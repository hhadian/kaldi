// chain/chain-training.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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
#include "chain/chain-denominator.h"
#include "chain/chain-num-graph.h"
#include "chain/chain-full-numerator.h"
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
  if (nnet_output_deriv)
    nnet_output_deriv->CopyFromMat(deriv);
  return true;
}


void ComputeChainObjfAndDeriv(const ChainTrainingOptions &opts,
                              const DenominatorGraph &den_graph,
                              const Supervision &supervision,
                              const CuMatrixBase<BaseFloat> &nnet_output,
                              BaseFloat *objf,
                              BaseFloat *l2_term,
                              BaseFloat *weight,
                              CuMatrixBase<BaseFloat> *nnet_output_deriv,
                              CuMatrixBase<BaseFloat> *xent_output_deriv) {
  *weight = supervision.weight * supervision.num_sequences *
      supervision.frames_per_sequence;
  BaseFloat num_logprob_weighted;
  if (nnet_output_deriv)
    nnet_output_deriv->SetZero();
  bool num_ok = true;
  if (!supervision.e2e) {
    NumeratorComputation numerator(supervision, nnet_output);
    // note: supervision.weight is included as a factor in the derivative from
    // the numerator object, and the logprob too.
    num_logprob_weighted = numerator.Forward();
    if (nnet_output_deriv) {
      numerator.Backward(nnet_output_deriv);
      if (opts.num_scale != 1.0) {
        nnet_output_deriv->ApplyPow(opts.num_scale);
        for (int32 r = 0; r < nnet_output_deriv->NumRows(); r++)
          nnet_output_deriv->Row(r).Scale(1.0 / nnet_output_deriv->Row(r).Sum());
          // nnet_output_deriv->ApplySoftMaxPerRow(*nnet_output_deriv);
      }
      if (xent_output_deriv)
        xent_output_deriv->CopyFromMat(*nnet_output_deriv);
    } else if (xent_output_deriv) {
      // this branch will be taken if xent_output_deriv but not
      // nnet_output_deriv is set- which could happen if you want to compute the
      // cross-entropy objective but not the derivatives.
      xent_output_deriv->SetZero();
      numerator.Backward(xent_output_deriv);
    }
  } else if (supervision.ali.NumRows() != 0) {
    num_logprob_weighted = -8 * *weight; //some fixed value-- not important
    num_ok = true;
    KALDI_LOG << "Pre-computed soft alignments available. Using them...";
    if (nnet_output_deriv) {
      Matrix<BaseFloat> derivs(supervision.ali.NumRows(), supervision.ali.NumCols(), kUndefined);
      supervision.ali.CopyToMat(&derivs);
      nnet_output_deriv->CopyFromMat(derivs);
    }
  } else {
    NumeratorGraph ng(supervision, opts.offset_first_transitions);
    if (!opts.pdf_map_filename.empty())
      ng.MapPdfs(opts.pdf_map);

    FullNumeratorComputation fnum(opts, ng, nnet_output);

    if (!opts.viterbi) {
      num_logprob_weighted = fnum.Forward();
      KALDI_LOG << "Doing Forward-Backward. Num Logprob: "
                << num_logprob_weighted / (*weight);
      num_ok = (num_logprob_weighted - num_logprob_weighted == 0);
      KALDI_LOG << "Num Forward "<< (num_ok ? "succeeded" : "failed") <<".";
      if (nnet_output_deriv && num_ok) {
        num_ok = fnum.Backward(supervision.weight, nnet_output_deriv);
        KALDI_LOG << "Num Backward " << (num_ok ? "succeeded" : "failed")
                  << " (dbl-chk: " << (opts.check_derivs ? "true" : "false") << ").";
      }
    } else {
      KALDI_LOG << "Doing Viterbi...";
      num_ok = fnum.Viterbi(supervision.weight, &num_logprob_weighted,
                            nnet_output_deriv);
      if (num_ok)
        KALDI_LOG << "Viterbi succeeded. Viterbi Logprob: "
                  << num_logprob_weighted / (*weight);
      else
        KALDI_LOG << "Viterbi failed. Viterbi Logprob: "
                  << num_logprob_weighted / (*weight);
    }
    if (!num_ok && opts.equal_align) {
      num_ok = TryEqualAlign(supervision, &num_logprob_weighted,
                             nnet_output_deriv);
      KALDI_LOG << "Doing EqualAlign. EqAlign Logprob: "
                << num_logprob_weighted / (*weight) << "     OK:" << num_ok;
    } else if (!num_ok && !opts.equal_align) {
      KALDI_LOG << "Not doing equal-align because it is disabled.";
    }
    if (xent_output_deriv && nnet_output_deriv)
      xent_output_deriv->CopyFromMat(*nnet_output_deriv);
  }

  if (!opts.write_trans_stats_prefix.empty() && nnet_output_deriv && num_ok) {
    // compute and save transition stats to a file with random name with prefix.
    // later all these stats will be aggregated and applied to transition probs
    // in the num and den graph
    // these are computed for pdf_id's not tid's actually
    // [so instead of applying them to num/den graph we can simply apply them
    // nnet output]
    CuVector<BaseFloat> stats(nnet_output_deriv->NumCols(), kUndefined);
    stats.AddRowSumMat(1.0, *nnet_output_deriv, 0.0);
    std::stringstream ss;
    ss << opts.write_trans_stats_prefix << time(0) << "-" << Rand() << ".stats";
    std::ofstream of(ss.str().c_str());
    stats.Write(of, false);
  }

  if (GetVerboseLevel() >= 2 && nnet_output_deriv && !num_ok) {
    // Save nnet-output and derivs on disk
    KALDI_LOG << "Saving nnet-output and nnet-output-deriv for debugging...";
    std::ofstream of1("nnet-out-deriv.mat");
    nnet_output_deriv->Write(of1, false);
    std::ofstream of2("nnet-out.mat");
    nnet_output.Write(of2, false);
    KALDI_LOG << "Saved.";
  }

  bool ok = true;
  BaseFloat den_logprob = 0.0;
  if (!opts.disable_mmi && num_ok) {
    DenominatorComputation denominator(opts, den_graph,
                                       supervision.num_sequences,
                                       nnet_output);

    den_logprob = denominator.Forward();
    KALDI_LOG << "Den Logprob: "
              << supervision.weight * den_logprob / (*weight);

    if (nnet_output_deriv) {
      ok = denominator.Backward(-supervision.weight,
                                nnet_output_deriv);
      KALDI_LOG << "Den Backward " << (ok ? "succeeded" : "failed") << ".";
    }
  } else if (opts.disable_mmi) {
    KALDI_LOG << "Not doing denominator because MMI is disabled.";
  } else {
    KALDI_LOG << "Not doing denominator because numerator computation failed.";
  }

  *objf = num_logprob_weighted - supervision.weight * den_logprob;
  KALDI_LOG << "Objf: " << *objf / *weight;

  if (!((*objf) - (*objf) == 0) || !ok || !num_ok) {
    // inf or NaN detected, or denominator computation returned false.
    if (nnet_output_deriv)
      nnet_output_deriv->SetZero();
    if (xent_output_deriv)
      xent_output_deriv->SetZero();
    BaseFloat default_objf = -10;
    KALDI_WARN << "Objective function is " << (*objf)
               << " and denominator computation (if done) returned "
               << std::boolalpha << ok << " " << std::boolalpha << num_ok
               << ", setting objective function to " << default_objf
               << " per frame.";
    *objf  = default_objf * *weight;
  }

  // This code helps us see how big the derivatives are, on average,
  // for different frames of the sequences.  As expected, they are
  // smaller towards the edges of the sequences (due to the penalization
  // of 'incorrect' pdf-ids.
  if (GetVerboseLevel() >= 1 && nnet_output_deriv != NULL) {
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
  } else if (num_ok) { // we should have some derivs to include a L2 term
    // compute the l2 penalty term and its derivative
    BaseFloat scale = supervision.weight * opts.l2_regularize;
    *l2_term = -0.5 * scale * TraceMatMat(nnet_output, nnet_output, kTrans);
    if (nnet_output_deriv)
      nnet_output_deriv->AddMat(-1.0 * scale, nnet_output);
  }
}


}  // namespace chain
}  // namespace kaldi
