// chain/chain-full-numerator.cc

// Copyright      2015   Hossein Hadian

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


#include "chain/chain-generic-numerator.h"
#include "chain/chain-kernels-ansi.h"

namespace kaldi {
namespace chain {

FullNumeratorComputation::FullNumeratorComputation(
                                    const ChainTrainingOptions &opts,
                                    const NumeratorGraph &num_graph,
                                    const CuMatrixBase<BaseFloat> &nnet_output):
    opts_(opts),
    num_graph_(num_graph),
    num_sequences_(num_graph.NumSequences()),
    frames_per_sequence_(nnet_output.NumRows() / num_sequences_),
    exp_nnet_output_transposed_(nnet_output, kTrans),
    nnet_output_deriv_transposed_(
        exp_nnet_output_transposed_.NumRows(),
        std::min<int32>(exp_nnet_output_transposed_.NumCols(),
                        static_cast<int32>(kMaxDerivTimeSteps) *
                        num_sequences_)),
    alpha_(frames_per_sequence_ + 1,
           num_graph_.MaxNumStates() * num_sequences_ + num_sequences_,
           kSetZero),
    // we actually do not need beta for state num_graph_.MaxNumStates(),
    // so no "+ num_sequences_"
    beta_(2, num_graph_.MaxNumStates() * num_sequences_,
          kSetZero),
    tot_prob_(num_sequences_, kUndefined),
    tot_log_prob_(num_sequences_, kUndefined),
    ok_(true) {

  seq_ok_.resize(num_sequences_, true);
  if (opts_.viterbi) {
    using std::vector;
    alpha_.Resize(0, 0);
    beta_.Resize(0, 0);
    logdelta_ = Matrix<BaseFloat>(frames_per_sequence_ + 1,
                                num_graph_.MaxNumStates() * num_sequences_,
                                kUndefined);
    logdelta_.Set(-std::numeric_limits<BaseFloat>::infinity());
    sai_ = vector<vector<vector<int32> > >(frames_per_sequence_ + 1,
                                           vector<vector<int32> >(num_sequences_,
                                                                  vector<int32>(num_graph_.MaxNumStates(), -1)));
  }
  KALDI_ASSERT(nnet_output.NumRows() % num_sequences_ == 0);
  if (!opts.trans_probs_filename.empty()) {
    KALDI_ASSERT(opts.trans_probs.Dim() == exp_nnet_output_transposed_.NumRows());
    // opts.trans_prob is in log space
    exp_nnet_output_transposed_.AddVecToCols(1.0, opts.trans_probs);
  }
  exp_nnet_output_transposed_.ApplyExp();

  for (int32 i = 0; i < opts.pdf_map.size(); i++) {
    if (opts.pdf_map[i] != i) {
      KALDI_LOG << "pdf_id " << i << " is mapped to " << opts.pdf_map[i];
      //some_pdf = i;
      break;
    }
  }
  tot_pruned_ = tot_states_ = tot_active_arcs_ = tot_arcs_ = 0;
}

// TODO: merge this with the current Forward/Backward functions --> more elagant
// Also handle Offset if enabled
bool FullNumeratorComputation::Viterbi(
    BaseFloat deriv_weight,
    BaseFloat *tot_logprob,
    CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  using std::vector;
  KALDI_ASSERT(opts_.viterbi);
  ok_ = true;

  // init
  BaseFloat *first_frame_logdelta = logdelta_.RowData(0);
  SubVector<BaseFloat> logdelta_state0(first_frame_logdelta, num_sequences_);
  logdelta_state0.Set(0.0);

  // viterbi
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows(),
      max_num_hmm_states = num_graph_.MaxNumStates();
  vector<vector<int32> > ali(num_sequences_, vector<int32>(frames_per_sequence_, -1));
  for (int32 t = 1; t <= frames_per_sequence_; t++) {

    SubMatrix<BaseFloat> this_logdelta(
        logdelta_.RowData(t),
        max_num_hmm_states,
        num_sequences_,
        num_sequences_);
    SubMatrix<BaseFloat> prev_logdelta(
        logdelta_.RowData(t - 1),
        max_num_hmm_states,
        num_sequences_,
        num_sequences_);
    // 'probs' is the matrix of pseudo-likelihoods for frame t - 1.
    SubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               (t - 1) * num_sequences_, num_sequences_);
    for (int32 seq = 0; seq < num_sequences_; seq++) {
      bool at_least_one_state_active = false;
      for (int32 j = 0; j < num_graph_.NumStates()[seq]; j++) {  // iterate over transitions i --> j
        float max = -std::numeric_limits<float>::infinity();
        int32 argmax = -1;
        for (int32 tr_idx = num_graph_.BackwardTransitions()[seq * max_num_hmm_states + j].first;
             tr_idx < num_graph_.BackwardTransitions()[seq * max_num_hmm_states + j].second; tr_idx++) {
          const DenominatorGraphTransition &tr = num_graph_.Transitions()[tr_idx];
          BaseFloat log_transition_prob = Log(tr.transition_prob);
          int32 pdf_id = tr.pdf_id, i = tr.hmm_state;
          BaseFloat log_prob = Log(probs(pdf_id, seq));
          BaseFloat this_arc = prev_logdelta(i, seq) + log_transition_prob + log_prob;
          if (this_arc > max) {
            max = this_arc;
            argmax = tr_idx;
          }
        } // i
        if (max - max == 0 && argmax != -1)
          at_least_one_state_active = true;
        this_logdelta(j, seq) = max;
        sai_[t - 1][seq][j] = argmax;
      } // j
      if (!at_least_one_state_active) {
        KALDI_WARN << "Viterbi failure: "
                   << " seq: " << seq
                   << " t: " << t;
        ok_ = false;
        return false;
      }

      // finalize
      if (t == frames_per_sequence_) {
        float max = -std::numeric_limits<float>::infinity();
        int32 argmax = -1;
        for (int j = 0; j < num_graph_.NumStates()[seq]; j++) {
          float fin_logprob = Log(num_graph_.FinalProbs()(j, seq));
          if (this_logdelta(j, seq) + fin_logprob > max) {
            max = this_logdelta(j, seq) + fin_logprob;
            argmax = sai_[t - 1][seq][j];
          }
        }
        if (max - max != 0 || argmax == -1) {
          KALDI_WARN << "Viterbi final state failure: "
                     << " seq: " << seq
                     << " t: " << t
                     << " max: " << max
                     << " argmax: " << argmax;
          ok_ = false;
          return false;
        }
        tot_log_prob_(seq) = max;
        ali[seq][t - 1] = num_graph_.Transitions()[argmax].pdf_id;
        // backtrack
        int32 tr_idx = argmax;
        for (int32 t1 = frames_per_sequence_ - 2; t1 >= 0; t1--) {
          tr_idx = sai_[t1][seq][num_graph_.Transitions()[tr_idx].hmm_state];
          ali[seq][t1] = num_graph_.Transitions()[tr_idx].pdf_id;
        }
      } //fin

    } // seq
  } // t


  // derivatives

  // free some space
  logdelta_.Resize(0, 0);
  *tot_logprob = tot_log_prob_.Sum();
  if (nnet_output_deriv) {
    Matrix<BaseFloat> deriv(nnet_output_deriv->NumRows(), nnet_output_deriv->NumCols(),
                            kSetZero);
    for (int32 s = 0; s < num_sequences_; s++)
      for (int32 t = 0; t < frames_per_sequence_; t++)
        deriv(t * num_sequences_ + s, ali[s][t]) = 1.0;
    nnet_output_deriv->CopyFromMat(deriv);
  }
  /*  if (GetVerboseLevel() >= 3) {
    for (int32 s = 0; s < num_sequences_; s++) {
      std::cout << "seq-" << s << "  " << tot_log_prob_(s) << "    ";
      for (int32 t = 0; t < frames_per_sequence_; t++)
        std::cout << ali[s][t] << " ";
      std::cout << "\n";
    }
    }*/
  return true;
}

void FullNumeratorComputation::AlphaFirstFrame() {
  // select alpha for time 0
  BFloat *first_frame_alpha = alpha_.RowData(0);
  // now make a view of the first num_sequences elements (i.e. alpha_0(0)
  // for all sequences)
  // initializer takes [pointer, length].
  SubVector<BFloat> alpha_hmm_state0(first_frame_alpha, num_sequences_);
  // set alpha_0(0) for all sequences to 1.0 and leave the rest to be 0.0.
  // i.e. the only start state is state 0.
  // alpha_hmm_state0.Set(1.0e-200);
  //  alpha_hmm_state0.Write(std::cout, false);
  alpha_hmm_state0.Set(1.0);
  // Now compute alpha-sums for t==0 which is obviously 1.0 for each sequence
  SubVector<BFloat> alpha_sum_vec(
                                     first_frame_alpha +
                                     num_graph_.MaxNumStates() * num_sequences_,
                                     num_sequences_);
  alpha_sum_vec.Set(1.0);
  // KALDI_LOG << "al-sum for seq=7, t=" << 0 << " is " << alpha_sum_vec(7);
}


// the alpha computation for some 0 < t <= num_time_steps_.
void FullNumeratorComputation::AlphaGeneralFrame(int32 t) {
  KALDI_ASSERT(t > 0 && t <= frames_per_sequence_);
  BFloat *this_alpha = alpha_.RowData(t);
  const BFloat *prev_alpha = alpha_.RowData(t - 1);
  const Int32Pair *backward_transitions = num_graph_.BackwardTransitions();
  const DenominatorGraphTransition *transitions = num_graph_.Transitions();
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows(),
      max_num_hmm_states = num_graph_.MaxNumStates(),
      num_sequences = num_sequences_;

  // 'probs' is the matrix of pseudo-likelihoods for frame t - 1.
  SubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               (t - 1) * num_sequences_, num_sequences_);
  const BaseFloat *prob_data = probs.Data();

  int32 prob_stride = probs.Stride();
  for (int32 s = 0; s < num_sequences; s++) {
    BFloat max_alpha = 0.0;
    BFloat inv_arbitrary_scale =
        prev_alpha[max_num_hmm_states * num_sequences + s];  //#SCC#
    for (int32 h = 0; h < num_graph_.NumStates()[s]; h++) {
      double this_tot_alpha = 0.0;
      const DenominatorGraphTransition
          *trans_iter = transitions +
            backward_transitions[s*max_num_hmm_states+h].first,
          *trans_end = transitions +
            backward_transitions[s*max_num_hmm_states+h].second;
      for (; trans_iter != trans_end; ++trans_iter) {
        BFloat transition_prob = trans_iter->transition_prob;
        int32 pdf_id = trans_iter->pdf_id,
              prev_hmm_state = trans_iter->hmm_state;
        BFloat prob = prob_data[pdf_id * prob_stride + s],
            this_prev_alpha = prev_alpha[prev_hmm_state * num_sequences + s];
        this_tot_alpha += this_prev_alpha / inv_arbitrary_scale * transition_prob * prob; //#SCC#

        if (this_tot_alpha - this_tot_alpha != 0) {
          //KALDI_LOG << "t: " << t << ", seq: " << s << ", h: " << h
          //<< ", prev-alpha: " << this_prev_alpha
          //<< ", prob: " << prob
          //<< ", inv_arbitrary_scale: " << inv_arbitrary_scale;
          //KALDI_ERR << "Beta failure.";
        }


      }
      // Let arbitrary_scale be the inverse of the alpha-sum value that we
      // store in the same place we'd store the alpha for the state numbered
      // 'max_num_hmm_states'. We multiply this into all the
      // transition-probabilities from the previous frame to this frame, in
      // both the forward and backward passes, in order to keep the alphas in
      // a good numeric range.  This won't affect the posteriors, but when
      // computing the total likelihood we'll need to compensate for it later
      // on.
      //KALDI_ASSERT(this_tot_alpha - this_tot_alpha == 0);
      this_alpha[h * num_sequences + s] = this_tot_alpha ;//* arbitrary_scale; //#SCC#
      if (this_tot_alpha > max_alpha)
        max_alpha = this_tot_alpha;
    }  // state h

    if (opts_.prune != 2 && opts_.prune != 3)
      continue;
    if (max_alpha == 0.0 || max_alpha - max_alpha != 0.0) {
      KALDI_LOG << "Bad max-alpha for seq " << s << " at time " << t << ": "
                << max_alpha;
      continue;
    }
    KALDI_VLOG(2) << "Seq[" << s << "]@" << t << "  -->  max-alpha: " << max_alpha << ", num-states: " << num_graph_.NumStates()[s];
    /// Prune
    BFloat max_alpha_log = Log(max_alpha);
    int32 npruned = 0;
    BFloat totpruned = 0.0;
    BFloat totnonpruned = 0.0;
    const Int32Pair *forward_transitions = num_graph_.ForwardTransitions();
    for (int32 h = 0; h < num_graph_.NumStates()[s]; h++) {
      if (Log(this_alpha[h * num_sequences + s]) + opts_.prune_beam < max_alpha_log) {
        npruned++;
        totpruned += this_alpha[h * num_sequences + s];
        // prune it
        if (opts_.prune == 1) {  // first method of pruning  -- not possibe --> we have different sequences sharing the same obs matrix
          // zero out all observations at time t for start-state = h
        } else if (opts_.prune == 2 || opts_.prune == 3) {  // do forward-pruning
          this_alpha[h * num_sequences + s] = 0.0;
        }
      } else {
        totnonpruned += this_alpha[h * num_sequences + s];
      }
    }
    KALDI_VLOG(1) << "Seq[" << s << "]@" << t << "  -->  pruned " << npruned
                  << " states with tot-prob: " << totpruned
                  << " remaining states: " << num_graph_.NumStates()[s] - npruned
                  << " and tot-unpruned-prob: " << totnonpruned;
    tot_pruned_ += npruned;
    tot_states_ += num_graph_.NumStates()[s];
  }   // seq s

  // Now compute alpha-sums for frame t:
  SubMatrix<BFloat> alpha_mat(this_alpha,
                              num_graph_.MaxNumStates(),
                              num_sequences_,
                              num_sequences_);
  if (t == frames_per_sequence_) {// last alpha
    Matrix<BFloat> dd(num_graph_.FinalProbs());
    alpha_mat.MulElements(dd);
  }
  SubVector<BFloat> alpha_sum_vec(this_alpha +
                                     num_graph_.MaxNumStates() * num_sequences_,
                                     num_sequences_);
  alpha_sum_vec.AddRowSumMat(1.0, alpha_mat, 0.0);
  /*if (GetVerboseLevel() >= 3) {
    std::cout << "Alpha(t=" << t << "): ";
    for (int i = 0; i < num_graph_.MaxNumStates(); i++)
      std::cout << alpha_mat(i, 0) << " ";
    std::cout << "\n";
  }
  if (GetVerboseLevel() >= 3) {
    std::cout << "Alphasum(t=" << t << "): ";
    for (int i = 0; i < num_graph_.MaxNumStates(); i++)
      std::cout << alpha_sum_vec(i) << " ";
    std::cout << "\n";
  }*/


  //std::cout << "alpha-sums for t = " << t << ": \n";
  //alpha_sum_vec.Write(std::cout, false);
  //  std::cout << "t = " << t << ": ";
  //  for (int s=0;s<=117;s++) std::cout << " " << alpha_mat(s, 0);
  //  std::cout<<"\n";
  //if (t == frames_per_sequence_) {
  //  const Matrix<BaseFloat> &x = num_graph_.FinalProbs();
  //  for (int i = 0; i < num_graph_.MaxNumStates(); i++)
  //    if (x(i, 0) != 0.0)
  //std::cout << "i=" << i << ": " << x(i, 0) << " and alpha(i,0)=" << alpha_mat(i, 0) << "\n";
  //  }
/*
  std::cout << "Alphas for seq 7, t= " << t << ":\n";
  for (int h = 0; h < num_graph_.NumStates()[7]; h++)
    std::cout << alpha_mat(h, 7) << " ";
  std::cout << std::endl;
*/
 //  KALDI_LOG << "al-sum for seq=7, t=" << t << " is " << alpha_sum_vec(7);
}

BaseFloat FullNumeratorComputation::Forward() {
  KALDI_ASSERT(!opts_.viterbi);
  AlphaFirstFrame();
  for (int32 t = 1; t <= frames_per_sequence_; t++) {
    AlphaGeneralFrame(t);
  }
  KALDI_LOG << "*************************** Forward pruning stats: "
            << " tot-pruned: " << tot_pruned_
            << " tot_states_: " << tot_states_
            << " ratio: " << 1.0 * tot_pruned_ / tot_states_;
  return ComputeTotLogLike();
}

BaseFloat FullNumeratorComputation::ComputeTotLogLike() {
  tot_prob_.Resize(num_sequences_);
  // View the last alpha as a matrix of size num-hmm-states by num-sequences.
  SubMatrix<BFloat> last_alpha(
      alpha_.RowData(frames_per_sequence_),
      num_graph_.MaxNumStates(),
      num_sequences_,
      num_sequences_);


  tot_prob_.AddRowSumMat(1.0, last_alpha, 0.0);
  if (tot_prob_.Sum() == 0.0) {  /// no token has reached the final states (it can happen when pruning is enabled)
    std::fill(seq_ok_.begin(), seq_ok_.end(), false);
    return -std::numeric_limits<BaseFloat>::infinity();
  }

  // we should probably add an ApplyLog() function that takes a vector argument.
  tot_log_prob_ = tot_prob_;
  tot_log_prob_.ApplyLog();
  if (num_graph_.AreFirstTransitionsScaled())
    tot_log_prob_.AddVec(1.0, num_graph_.FirstTransitionOffsets());

  // BFloat tot_log_prob = tot_log_prob_.Sum();

  int32 max_num_hmm_states = num_graph_.MaxNumStates();
  SubMatrix<BFloat> inv_arbitrary_scales(
      alpha_, 0, frames_per_sequence_,
      num_sequences_ * max_num_hmm_states, num_sequences_);
  Matrix<BFloat> log_inv_arbitrary_scales(
      inv_arbitrary_scales);
  log_inv_arbitrary_scales.ApplyLog();


  //  BFloat log_inv_arbitrary_scales_product =
  //      log_inv_arbitrary_scales.Sum();
  Vector<BFloat> totscales_vect(num_sequences_);
  totscales_vect.AddRowSumMat(1.0, log_inv_arbitrary_scales, 0.0);
  BFloat tot_log_prob = 0.0;
  for (int32 i = 0; i < num_sequences_; i++) {
    tot_log_prob_(i) += totscales_vect(i);
    if (tot_log_prob_(i) - tot_log_prob_(i) == 0.0)
      tot_log_prob += tot_log_prob_(i);
    else
      seq_ok_[i] = false;
  }

    //  return tot_log_prob + log_inv_arbitrary_scales_product;
  return tot_log_prob;
}


bool FullNumeratorComputation::Backward(
    BaseFloat deriv_weight,
    CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  KALDI_ASSERT(!opts_.viterbi);
  BetaLastFrame();
  for (int32 t = frames_per_sequence_ - 1; t >= 0; t--) {
    BetaGeneralFrame(t);
    // if pruning is enabled we need to normalize the occupation probs:
    if (opts_.prune == 2 || opts_.prune == 3 ||  opts_.prune == 4) {
      KALDI_VLOG(2) << "Time: " << t << ". Normalizing...";
      int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps),
          num_pdfs = exp_nnet_output_transposed_.NumRows();
      SubMatrix<BaseFloat>   this_log_prob_deriv(nnet_output_deriv_transposed_, 0, num_pdfs,
                                                 t_wrapped * num_sequences_, num_sequences_);
      Matrix<BaseFloat> this_log_prob_deriv_trans(this_log_prob_deriv, kTrans);
      for (int32 seq = 0; seq < num_sequences_; seq++) {
        BFloat sum = this_log_prob_deriv_trans.Row(seq).Sum();
        KALDI_VLOG(2) << "\t Seq[" << seq << "] --> sum: " << sum << ", seq-ok: " << seq_ok_[seq];
        if (sum - sum == 0.0 && seq_ok_[seq]) {  // F-B has been successful for this sequence
          this_log_prob_deriv_trans.Row(seq).Scale(1.0 / sum);
        } else {
          seq_ok_[seq] = false;
        }
      }
      this_log_prob_deriv.CopyFromMat(this_log_prob_deriv_trans, kTrans);
    }

    if (GetVerboseLevel() >= 1 || t == 0 || (t == frames_per_sequence_ - 1 && opts_.check_derivs))
      BetaGeneralFrameDebug(t);
    if (t % kMaxDerivTimeSteps == 0) {
      // commit the derivative stored in exp_nnet_output_transposed_ by adding
      // its transpose to the appropriate sub-matrix of 'nnet_output_deriv'.
      int32 chunk_frames = std::min<int32>(static_cast<int32>(kMaxDerivTimeSteps),
                                           frames_per_sequence_ - t),
                num_pdfs = exp_nnet_output_transposed_.NumRows();
      SubMatrix<BaseFloat> transposed_deriv_part(
          nnet_output_deriv_transposed_,
          0, num_pdfs,
          0, chunk_frames * num_sequences_);
      CuMatrix<BaseFloat> tmp(transposed_deriv_part);
      CuSubMatrix<BaseFloat> output_deriv_part(
          *nnet_output_deriv,
          t * num_sequences_, chunk_frames * num_sequences_,
          0, num_pdfs);
      output_deriv_part.AddMat(deriv_weight, tmp, kTrans);
      if (t != 0)
        transposed_deriv_part.SetZero();
    }
  }
//  nnet_output_deriv->AddMat(
//                           deriv_weight, nnet_output_deriv_transposed_, kTrans);

  return ok_;
}

void FullNumeratorComputation::BetaLastFrame() {
  // sets up the beta quantity on the last frame (frame ==
  // frames_per_sequence_).  Note that the betas we use here contain a
  // 1/(tot-prob) factor in order to simplify the backprop.

  int32 t = frames_per_sequence_;
  BFloat *last_frame_beta = beta_.RowData(t % 2);

  // create a 'fake matrix' - view this row as a matrix.
  SubMatrix<BFloat> beta_mat(last_frame_beta,
                                num_graph_.MaxNumStates(),
                                num_sequences_,
                                num_sequences_);

  // There is only 1 final state in each sequence's HMM, and its prob is 1.0
  // Please refer to chain-supervision.h,cc for more info
  // since final state indexes are different for each sequence, we set them in
  // a for loop.
  /*int32 *num_states_cpu = new int32[num_graph_.NumSequences()];
  num_graph_.CopyNumStatesToCpu(num_states_cpu);  //TODO(hhadian) this might be really slow -- check it
  for (int32 seq = 0; seq < num_sequences_; seq++) {
    int32 final_state = num_states_cpu[seq] - 1;
    beta_mat(final_state, seq) = 1.0 / tot_prob_(seq);
  }
  delete num_states_cpu;*/

  Vector<BFloat> inv_tot_prob(tot_prob_);
  //if (GetVerboseLevel() >= 2) {
  //  std::cout << "tot_prob: ";
  //  inv_tot_prob.Write(std::cout, false);
  //}
  inv_tot_prob.InvertElements();
  beta_mat.CopyRowsFromVec(inv_tot_prob);
  Matrix<BFloat> dd(num_graph_.FinalProbs());
  beta_mat.MulElements(dd);
//  const Matrix<BaseFloat> &x = num_graph_.FinalProbs();
//  for (int i = 0; i < num_graph_.MaxNumStates(); i++)
//    std::cout << x(i, 0) << " ";
  /*if (GetVerboseLevel() >= 3) {
    std::cout << "Beta(T): ";
    for (int i = 0; i < num_graph_.MaxNumStates(); i++)
      std::cout << beta_mat(i, 0) << " ";
    std::cout << "\n";
    }*/
}

void FullNumeratorComputation::BetaGeneralFrame(int32 t) {
  KALDI_ASSERT(t >= 0 && t < frames_per_sequence_);
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows();
  // t_wrapped gives us the time-index we use when indexing
  // nnet_output_deriv_transposed_; to save memory we limit the size of the
  // matrix, storing only chunks of frames at a time, and we add it to the
  // non-transposed output whenever we finish a chunk.
  int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps);
  BFloat *this_alpha = alpha_.RowData(t),
                       *next_beta = beta_.RowData((t + 1) % 2);
  BFloat *this_beta = beta_.RowData(t % 2);
  const Int32Pair *forward_transitions = num_graph_.ForwardTransitions();
  const DenominatorGraphTransition *transitions = num_graph_.Transitions();
  // 'probs' is the matrix of pseudo-likelihoods for frame t.
  SubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                             t * num_sequences_, num_sequences_),
      log_prob_deriv(nnet_output_deriv_transposed_, 0, num_pdfs,
                     t_wrapped * num_sequences_, num_sequences_);

  int32 max_num_hmm_states = num_graph_.MaxNumStates(),
        num_sequences = num_sequences_;


  int32 prob_stride = probs.Stride(),
        deriv_stride = log_prob_deriv.Stride();
  const BaseFloat *prob_data = probs.Data();
  BaseFloat *log_prob_deriv_data = log_prob_deriv.Data();
  for (int32 s = 0; s < num_sequences; s++) {
    if (!seq_ok_[s])
      continue;
    BaseFloat max_beta = 0.0;
    for (int32 h = 0; h < num_graph_.NumStates()[s]; h++) {
      BaseFloat this_alpha_prob = this_alpha[h * num_sequences + s],
          inv_arbitrary_scale =
          this_alpha[max_num_hmm_states * num_sequences + s];
      double tot_variable_factor = 0.0;
      //BFloat occupation_factor = this_alpha_prob /
      //    inv_arbitrary_scale;
      const DenominatorGraphTransition
          *trans_iter = transitions +
            forward_transitions[s*max_num_hmm_states + h].first,
          *trans_end = transitions +
            forward_transitions[s*max_num_hmm_states + h].second;
      for (; trans_iter != trans_end; ++trans_iter) {
        BaseFloat transition_prob = trans_iter->transition_prob;
        int32 pdf_id = trans_iter->pdf_id,
            next_hmm_state = trans_iter->hmm_state;
        BFloat variable_factor = transition_prob *
            next_beta[next_hmm_state * num_sequences + s] *
            prob_data[pdf_id * prob_stride + s] / inv_arbitrary_scale;  // #SCC#
        tot_variable_factor += variable_factor;
        BFloat occupation_prob = variable_factor * this_alpha_prob; //occupation_factor; #SCC#
        log_prob_deriv_data[pdf_id * deriv_stride + s] += occupation_prob;


        //if (variable_factor - variable_factor != 0  || tot_variable_factor - tot_variable_factor != 0) {
          //KALDI_LOG << "t: " << t << ", seq: " << s << ", h: " << h << ", pdf_id: " << pdf_id
          //<< ", var-factor: " << variable_factor
          //<< ", tot-var-factor: " << tot_variable_factor
          //<< ", trans-prob: " << transition_prob
          //<< ", ocup-prob: " << occupation_prob
          //<< ", next-beta: " << next_beta[next_hmm_state * num_sequences + s]
          //<< ", this_alpha_prob: " << this_alpha_prob
          //<< ", obs-prob: " << prob_data[pdf_id * prob_stride + s]
          //<< ", inv arbitrary_scale: " << inv_arbitrary_scale;
          //KALDI_ERR << "Beta failure.";
        //}


      }
      this_beta[h * num_sequences + s] =
          tot_variable_factor ; /// inv_arbitrary_scale; #SCC#
      if (tot_variable_factor > max_beta)
        max_beta = tot_variable_factor;
    }  // h


    if (opts_.prune != 3 && opts_.prune != 4)
      continue;
    if (max_beta == 0.0 || max_beta - max_beta != 0.0) {
      KALDI_LOG << "Bad max-beta for seq " << s << " at time " << t << ": "
                << max_beta;
      continue;
    }
    KALDI_VLOG(2) << "Seq[" << s << "]@" << t << "  -->  max-beta: " << max_beta
                  << ", num-states: " << num_graph_.NumStates()[s];
    /// Prune
    BFloat max_beta_log = Log(max_beta);
    int32 npruned = 0;
    BFloat totpruned = 0.0;
    BFloat totnonpruned = 0.0;
    for (int32 h = 0; h < num_graph_.NumStates()[s]; h++) {
      if (Log(this_beta[h * num_sequences + s]) + opts_.prune_beam < max_beta_log) {
        npruned++;
        totpruned += this_beta[h * num_sequences + s];
        // prune it
        if (opts_.prune == 1) {  // first method of pruning  -- not possibe --> we have different sequences sharing the same obs matrix
          // zero out all observations at time t for start-state = h
        } else if (opts_.prune == 3 || opts_.prune == 4) {  // do forward-pruning
          this_beta[h * num_sequences + s] = 0.0;
        }
      } else {
        totnonpruned += this_beta[h * num_sequences + s];
      }
    }
    KALDI_VLOG(1) << "Beta Seq[" << s << "]@" << t << "  -->  pruned " << npruned
                  << " states with tot-prob: " << totpruned
                  << " remaining states: " << num_graph_.NumStates()[s] - npruned
                  << " and tot-unpruned-prob: " << totnonpruned;



  }  // s
  SubMatrix<BFloat> beta_mat(this_beta,
                             num_graph_.MaxNumStates(),
                             num_sequences_,
                             num_sequences_);
  /*if (GetVerboseLevel() >= 3) {
    std::cout << "Beta(t=" << t << "): ";
    for (int i = 0; i < num_graph_.MaxNumStates(); i++)
      std::cout << beta_mat(i, 0) << " ";
    std::cout << "\n";
  }
  if (GetVerboseLevel() >= 2) {
    std::cout << "Deriv(t=" << t << "): ";
    for (int i = 0; i < num_pdfs; i++)
      std::cout << log_prob_deriv_data[i * deriv_stride] << " ";
    std::cout << "\n";
    }*/
}

void FullNumeratorComputation::BetaGeneralFrameDebug(int32 t) {
  int32 max_num_hmm_states = num_graph_.MaxNumStates();
  SubMatrix<BFloat> this_alpha(
      alpha_.RowData(t),
      max_num_hmm_states,
      num_sequences_,
      num_sequences_);
  SubMatrix<BFloat> this_beta(
      beta_.RowData(t % 2),
      max_num_hmm_states,
      num_sequences_,
      num_sequences_);
  Matrix<BFloat> alphabeta(this_alpha);
  alphabeta.MulElements(this_beta);
  Vector<BFloat> alphabeta_sums(num_sequences_);
  alphabeta_sums.AddRowSumMat(1.0, alphabeta, 0.0);

  int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps),
        num_pdfs = exp_nnet_output_transposed_.NumRows();
  SubMatrix<BaseFloat> this_occup_probs(
      nnet_output_deriv_transposed_, 0, num_pdfs,
      t_wrapped * num_sequences_, num_sequences_);

  Vector<BaseFloat> occup_probs_sum(num_sequences_);
  occup_probs_sum.AddRowSumMat(1.0, this_occup_probs, 0.0);

  if (!(opts_.prune == 2 || opts_.prune == 3 || opts_.prune == 4)) {  // in these two cases of pruning, these won't hold!
    // do the cheks on a per-sequence basis
    for (int32 seq = 0; seq < num_sequences_; seq++)
      if (!ApproxEqual(alphabeta_sums(seq), 1.0)) {
        KALDI_WARN << "On time " << t
                   << " for seq" << seq << ", alpha-beta product "
                   << alphabeta_sums(seq) << " != 1.0";
        if (fabs(alphabeta_sums(seq) - 1.0) > 0.05 || alphabeta_sums(seq) - alphabeta_sums(seq) != 0) {
          KALDI_WARN << "Excessive error detected, will abandon seq " << seq;
          ok_ = false;
          seq_ok_[seq] = false;
        }
      }
  }
  // use higher tolerance, since we are using randomized pruning for the
  // log-prob derivatives.
  for (int32 seq = 0; seq < num_sequences_; seq++) {
    BFloat osum = occup_probs_sum(seq);
    if (!ApproxEqual(osum, 1.0, 0.01) && seq_ok_[seq]) {
      KALDI_WARN << "On time " << t << " for seq" << seq << ", log-prob-deriv sum "
                 << osum << " != " << 1.0;
      if (fabs(osum - 1.0) > 0.05 || osum - osum != 0) {
        KALDI_WARN << "Excessive error detected, will abandon seq " << seq;
        ok_ = false;
        seq_ok_[seq] = false;
      }
    }
  }

}


bool Align(const Supervision& sup,
           const MatrixBase<BaseFloat> &nnet_output_transposed,
           std::vector<std::vector<int32> > *alignments,
           BaseFloat *tot_logprob) {
  using std::vector;
  int32 B = sup.num_sequences,
      T = sup.frames_per_sequence;
  KALDI_ASSERT(sup.e2e_fsts.size() == B);
  struct InTransition {
    BaseFloat logprob;
    int32 pdf_id;
    int32 tid;
    int32 start_state;
  };
  InTransition null_trans;
  null_trans.logprob = -1; null_trans.pdf_id = -1; null_trans.tid = -1; null_trans.start_state = -1;
  // init
  // indexing order: seq, time, state
  vector<vector<vector<BaseFloat> > > logdelta(B, vector<vector<BaseFloat> >(T + 1));
  vector<vector<vector<InTransition> > > sai(B, vector<vector<InTransition> >(T + 1));
  for (int32 seq = 0; seq < B; seq++) {
    for (int32 t = 0; t < T + 1; t++) {
      BaseFloat value = (t == 0) ? 0 : -std::numeric_limits<BaseFloat>::infinity();
      logdelta[seq][t] = vector<BaseFloat>(sup.e2e_fsts[seq].NumStates(), value);
      sai[seq][t] = vector<InTransition>(sup.e2e_fsts[seq].NumStates(), null_trans);
    }
  }

  // init incmoing transitions for easy access
  vector<vector<vector<InTransition> > > in_transitions(B); // indexed by seq, state
  vector<vector<BaseFloat> > final_logprobs(B); // indexed by seq, state
  for (int32 seq = 0; seq < B; seq++) {
    in_transitions[seq] = vector<vector<InTransition> >(sup.e2e_fsts[seq].NumStates());
    final_logprobs[seq] = vector<BaseFloat>(sup.e2e_fsts[seq].NumStates(),
                                            -std::numeric_limits<BaseFloat>::infinity());
  }

  for (int32 seq = 0; seq < B; seq++) {
    for (int32 s = 0; s < sup.e2e_fsts[seq].NumStates(); s++) {
      final_logprobs[seq][s] = -sup.e2e_fsts[seq].Final(s).Value();
      for (fst::ArcIterator<fst::StdVectorFst> aiter(sup.e2e_fsts[seq], s);
           !aiter.Done();
           aiter.Next()) {
        const fst::StdArc &arc = aiter.Value();
        InTransition transition;
        transition.logprob = -arc.weight.Value();
        transition.pdf_id = arc.ilabel - 1;
        transition.tid = arc.olabel;
        transition.start_state = s;
        in_transitions[seq][arc.nextstate].push_back(transition);
      }
    }
  }

  // viterbi
  int32 num_pdfs = nnet_output_transposed.NumRows();
  KALDI_ASSERT(num_pdfs == sup.label_dim);
  vector<vector<int32> > ali(B, vector<int32>(T, -1));
  for (int32 t = 1; t <= T; t++) {
    // 'logprobs' is the matrix of pseudo-loglikelihoods for frame t - 1.
    SubMatrix<BaseFloat> logprobs(nnet_output_transposed, 0, num_pdfs, (t - 1) * B, B);
    for (int32 seq = 0; seq < B; seq++) {
      bool at_least_one_state_active = false;

      for (int32 j = 0; j < sup.e2e_fsts[seq].NumStates(); j++) {  // iterate over transitions i --> j
        float max = -std::numeric_limits<float>::infinity();
        InTransition argmax(null_trans);
        for (auto tr = in_transitions[seq][j].begin(); tr != in_transitions[seq][j].end(); tr++) {
          int32 i = tr->start_state;
          BaseFloat this_arc = logdelta[seq][t-1][i] + tr->logprob + logprobs(tr->pdf_id, seq);
          if (this_arc > max) {
            max = this_arc;
            argmax = *tr;
          }
        } // i
        if (max - max == 0)
          at_least_one_state_active = true;
        logdelta[seq][t][j] = max;
        sai[seq][t - 1][j] = argmax;
      } // j

      if (!at_least_one_state_active) {
        KALDI_WARN << "Viterbi failure: "
                   << " seq: " << seq
                   << " t: " << t;
        return false;
      }

      // finalize
      if (t == T) {
        float max = -std::numeric_limits<float>::infinity();
        InTransition argmax(null_trans);
        for (int j = 0; j < sup.e2e_fsts[seq].NumStates(); j++) {
          float fin_logprob = final_logprobs[seq][j];
          if (logdelta[seq][t][j] + fin_logprob > max) {
            max = logdelta[seq][t][j] + fin_logprob;
            argmax = sai[seq][t - 1][j];
          }
        }
        if (max - max != 0) {
          KALDI_WARN << "Viterbi final state failure: "
                     << " seq: " << seq
                     << " t: " << t
                     << " max: " << max;
          return false;
        }
        // tot_log_prob_(seq) = max;
        ali[seq][T - 1] = argmax.tid;
        // backtrack
        InTransition tr = argmax;
        for (int32 t1 = T - 2; t1 >= 0; t1--) {
          tr = sai[seq][t1][tr.start_state];
          ali[seq][t1] = tr.tid;
        }
      } //fin

    } // seq
  } // t

  if (alignments)
    *alignments = ali;
  return true;
}


}  // namespace chain
}  // namespace kaldi
