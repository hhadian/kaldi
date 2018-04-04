// chain/chain-training.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)
//                2018   Hossein Hadian

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "chain/chain-training.h"
#include "chain/chain-kernels-ansi.h"
#include "chain/chain-numerator.h"
#include "chain/chain-generic-numerator.h"
#include "chain/chain-denominator.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"

namespace kaldi {
namespace chain {

bool TryEqualAlign(const Supervision &supervision, BaseFloat *objf,
                   CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  typedef kaldi::int32 int32;
  using fst::SymbolTable;
  using fst::VectorFst;
  using fst::StdArc;
  int32 rand_seed = 27;
  int32 T = supervision.frames_per_sequence;
  int32 B = supervision.num_sequences;
  int32 N = supervision.label_dim;
  *objf = 0.0;
  Matrix<BaseFloat> deriv;
  if (nnet_output_deriv)
    deriv.Resize(nnet_output_deriv->NumRows(), nnet_output_deriv->NumCols(),
                 kSetZero);
  for (int32 s = 0; s < B; s++) {
    VectorFst<StdArc> path;
    if (EqualAlign(supervision.e2e_fsts[s], T, rand_seed, &path) ) {
      std::vector<int32> aligned_seq; // the labels are PdfIds + 1
      StdArc::Weight w;
      GetLinearSymbolSequence(path, &aligned_seq, (std::vector<int32> *)NULL, &w);
      KALDI_ASSERT(aligned_seq.size() == T);
      *objf -= w.Value();
      if (nnet_output_deriv) {
        for (int32 t = 0; t < T; t++)
          deriv(t*B + s, aligned_seq[t] - 1) = 1.0;
      }
    } else {
      KALDI_WARN << "AlignEqual: failed on seq: " << s;
      return false;
    }
  }
  if (nnet_output_deriv) {
    CuMatrix<BaseFloat> tmp(deriv);
    nnet_output_deriv->AddMat(1.0, tmp);
  }
  return true;
}


void ComputeChainObjfAndDerivE2e(const ChainTrainingOptions &opts,
                                 const DenominatorGraph &den_graph,
                                 const Supervision &supervision,
                                 const CuMatrixBase<BaseFloat> &nnet_output,
                                 BaseFloat *objf,
                                 BaseFloat *l2_term,
                                 BaseFloat *weight,
                                 CuMatrixBase<BaseFloat> *nnet_output_deriv,
                                 CuMatrix<BaseFloat> *xent_output_deriv) {
  BaseFloat num_logprob_weighted, den_logprob_weighted;
  bool denominator_ok = true;
  bool numerator_ok = true;
  *weight = supervision.weight * supervision.num_sequences *
      supervision.frames_per_sequence;

  if (nnet_output_deriv != NULL)
    nnet_output_deriv->SetZero();

  { // Doing the denominator first helps to reduce the maximum
    // memory use, as we can set 'xent_deriv' to nonempty after
    // we've freed the memory in this object.
    DenominatorComputation denominator(opts, den_graph,
                                       supervision.num_sequences,
                                       nnet_output);

    den_logprob_weighted = supervision.weight * denominator.Forward();

    KALDI_LOG << "Den Logprob per frame: "
              << den_logprob_weighted / (*weight);
    denominator_ok = (den_logprob_weighted - den_logprob_weighted == 0);
    if (denominator_ok && nnet_output_deriv) {
      denominator_ok = denominator.Backward(-supervision.weight,
                                nnet_output_deriv);
      KALDI_LOG << "Den Backward " << (denominator_ok ? "succeeded" : "failed") << ".";
    }

  }

  if (xent_output_deriv != NULL) {
    // the reason for kStrideEqualNumCols is so that we can share the memory
    // block with the memory that was used for exp_nnet_output_transposed_ from
    // chain-denominator.cc, which has just been freed; it also uses the
    // kStrideEqualNumCols arg (its shape is the transpose of this matrix's
    // shape).
    xent_output_deriv->Resize(nnet_output.NumRows(), nnet_output.NumCols(),
                              kSetZero, kStrideEqualNumCols);
  }

  if (!denominator_ok) {
    nnet_output_deriv->SetZero();
    den_logprob_weighted = -8.0 * (*weight);
  }
  {
    GenericNumeratorComputation numerator(supervision, nnet_output);
    // note: supervision.weight is included as a factor in the derivative from
    // the numerator object, as well as the returned logprob.
    num_logprob_weighted = numerator.Forward();
    KALDI_LOG << "Numerator logprob per frame: "
                  << num_logprob_weighted / (*weight);
    numerator_ok = (num_logprob_weighted - num_logprob_weighted == 0);
    if (!numerator_ok)
      KALDI_LOG << "Numerator forward failed.";

    if (xent_output_deriv && numerator_ok) {
      numerator_ok = numerator.Backward(xent_output_deriv);
      if (!numerator_ok)
        KALDI_LOG << "Numerator backward failed.";
      if (opts.num_scale != 1.0) {
        xent_output_deriv->ApplyPow(opts.num_scale);
        xent_output_deriv->ApplySoftMaxPerRow(*xent_output_deriv);
      }
      if (nnet_output_deriv)
        nnet_output_deriv->AddMat(1.0, *xent_output_deriv);
    } else if (nnet_output_deriv && numerator_ok) {
      numerator_ok = numerator.Backward(nnet_output_deriv);
      if (!numerator_ok)
        KALDI_LOG << "Numerator backward failed.";
    }
  }

  if (!numerator_ok && opts.equal_align) {
    numerator_ok = TryEqualAlign(supervision, &num_logprob_weighted,
                                 nnet_output_deriv);
    KALDI_LOG << "Doing EqualAlign. EqAlign Logprob: "
              << num_logprob_weighted / (*weight) << "     OK:" << numerator_ok;
  } else if (!numerator_ok && !opts.equal_align) {
    KALDI_LOG << "Not doing equal-align because it is disabled.";
  }
  if (xent_output_deriv && nnet_output_deriv)
    xent_output_deriv->CopyFromMat(*nnet_output_deriv);


  *objf = num_logprob_weighted - den_logprob_weighted;
  KALDI_LOG << "Objf: " << *objf / *weight;

  if ((*objf - *objf) != 0.0 || !denominator_ok || !numerator_ok) {
    // inf or NaN detected, or denominator computation returned false.
    if (nnet_output_deriv)
      nnet_output_deriv->SetZero();
    if (xent_output_deriv)
      xent_output_deriv->SetZero();
    BaseFloat default_objf = -10;
    KALDI_WARN << "Objective function is " << (*objf)
               << " and denominator computation (if done) returned "
               << std::boolalpha << denominator_ok
               << " and numerator computation returned "
               << std::boolalpha << numerator_ok
               << ", setting objective function to " << default_objf
               << " per frame.";
    *objf  = default_objf * *weight;
  }

  // This code helps us see how big the derivatives are, on average,
  // for different frames of the sequences.  As expected, they are
  // smaller towards the edges of the sequences (due to the penalization
  // of 'incorrect' pdf-ids.
  if (GetVerboseLevel() >= 1 && nnet_output_deriv != NULL && RandInt(0, 10) == 0) {
    int32 tot_frames = nnet_output_deriv->NumRows(),
 frames_per_sequence = supervision.frames_per_sequence,
       num_sequences = supervision.num_sequences;
    CuVector<BaseFloat> row_products(tot_frames);
    row_products.AddDiagMat2(1.0, *nnet_output_deriv, kNoTrans, 0.0);
    Vector<BaseFloat> row_products_cpu(row_products);
    Vector<BaseFloat> row_products_per_frame(frames_per_sequence);
    for (int32 i = 0; i < tot_frames; i++)
      row_products_per_frame(i / num_sequences) += row_products_cpu(i);
    KALDI_LOG << "Derivs per frame are " << row_products_per_frame;
  }

  *l2_term = 0.0;
  if (opts.l2_regularize != 0.0 && numerator_ok) {  // we should have some derivs to include a L2 term
    // compute the l2 penalty term and its derivative
    BaseFloat scale = supervision.weight * opts.l2_regularize;
    *l2_term = -0.5 * scale * TraceMatMat(nnet_output, nnet_output, kTrans);
    if (nnet_output_deriv)
      nnet_output_deriv->AddMat(-1.0 * scale, nnet_output);
  }
}


void ComputeChainObjfAndDeriv(const ChainTrainingOptions &opts,
                              const DenominatorGraph &den_graph,
                              const Supervision &supervision,
                              const CuMatrixBase<BaseFloat> &nnet_output,
                              BaseFloat *objf,
                              BaseFloat *l2_term,
                              BaseFloat *weight,
                              CuMatrixBase<BaseFloat> *nnet_output_deriv,
                              CuMatrix<BaseFloat> *xent_output_deriv) {
  if (supervision.e2e) {
    ComputeChainObjfAndDerivE2e(opts, den_graph, supervision,
                                nnet_output, objf, l2_term,
                                weight, nnet_output_deriv, xent_output_deriv);
    return;
  }

  BaseFloat num_logprob_weighted, den_logprob_weighted;
  bool ok = true;
  if (nnet_output_deriv != NULL)
    nnet_output_deriv->SetZero();

  { // Doing the denominator first helps to reduce the maximum
    // memory use, as we can set 'xent_deriv' to nonempty after
    // we've freed the memory in this object.
    DenominatorComputation denominator(opts, den_graph,
                                       supervision.num_sequences,
                                       nnet_output);

    den_logprob_weighted = supervision.weight * denominator.Forward();
    if (nnet_output_deriv)
      ok = denominator.Backward(-supervision.weight,
                                nnet_output_deriv);
  }

  if (xent_output_deriv != NULL) {
    // the reason for kStrideEqualNumCols is so that we can share the memory
    // block with the memory that was used for exp_nnet_output_transposed_ from
    // chain-denominator.cc, which has just been freed; it also uses the
    // kStrideEqualNumCols arg (its shape is the transpose of this matrix's
    // shape).
    xent_output_deriv->Resize(nnet_output.NumRows(), nnet_output.NumCols(),
                              kSetZero, kStrideEqualNumCols);
  }


  {
    NumeratorComputation numerator(supervision, nnet_output);
    // note: supervision.weight is included as a factor in the derivative from
    // the numerator object, as well as the returned logprob.
    num_logprob_weighted = numerator.Forward();

    if (xent_output_deriv) {
      numerator.Backward(xent_output_deriv);
      if (nnet_output_deriv)
        nnet_output_deriv->AddMat(1.0, *xent_output_deriv);
    } else if (nnet_output_deriv) {
      numerator.Backward(nnet_output_deriv);
    }
  }

  *objf = num_logprob_weighted - den_logprob_weighted;
  *weight = supervision.weight * supervision.num_sequences *
      supervision.frames_per_sequence;
  if (!((*objf) - (*objf) == 0) || !ok) {
    // inf or NaN detected, or denominator computation returned false.
    if (nnet_output_deriv)
      nnet_output_deriv->SetZero();
    if (xent_output_deriv)
      xent_output_deriv->SetZero();
    BaseFloat default_objf = -10;
    KALDI_WARN << "Objective function is " << (*objf)
               << " and denominator computation (if done) returned "
               << std::boolalpha << ok
               << ", setting objective function to " << default_objf
               << " per frame.";
    *objf  = default_objf * *weight;
  }

  // This code helps us see how big the derivatives are, on average,
  // for different frames of the sequences.  As expected, they are
  // smaller towards the edges of the sequences (due to the penalization
  // of 'incorrect' pdf-ids.
  if (GetVerboseLevel() >= 1 && nnet_output_deriv != NULL && RandInt(0, 10) == 0) {
    int32 tot_frames = nnet_output_deriv->NumRows(),
 frames_per_sequence = supervision.frames_per_sequence,
       num_sequences = supervision.num_sequences;
    CuVector<BaseFloat> row_products(tot_frames);
    row_products.AddDiagMat2(1.0, *nnet_output_deriv, kNoTrans, 0.0);
    Vector<BaseFloat> row_products_cpu(row_products);
    Vector<BaseFloat> row_products_per_frame(frames_per_sequence);
    for (int32 i = 0; i < tot_frames; i++)
      row_products_per_frame(i / num_sequences) += row_products_cpu(i);
    KALDI_LOG << "Derivs per frame are " << row_products_per_frame;
  }

  if (opts.l2_regularize == 0.0) {
    *l2_term = 0.0;
  } else {
    // compute the l2 penalty term and its derivative
    BaseFloat scale = supervision.weight * opts.l2_regularize;
    *l2_term = -0.5 * scale * TraceMatMat(nnet_output, nnet_output, kTrans);
    if (nnet_output_deriv)
      nnet_output_deriv->AddMat(-1.0 * scale, nnet_output);
  }
}


}  // namespace chain
}  // namespace kaldi
