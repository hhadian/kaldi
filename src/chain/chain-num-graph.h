// chain/chain-num-graph.h

// Copyright       2015  Hossein Hadian


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


#ifndef KALDI_CHAIN_CHAIN_NUM_GRAPH_H_
#define KALDI_CHAIN_CHAIN_NUM_GRAPH_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/kaldi-vector.h"
#include "chain/chain-datastruct.h"
#include "hmm/transition-model.h"
#include "chain/chain-supervision.h"

namespace kaldi {
namespace chain {


class NumeratorGraph {
 public:

  // This relates to #pdf_tying
  void MapPdfs(const std::vector<int32>& pdf_map) {
    // KALDI_ASSERT(pdf_map.size() == num_pdfs_);
    for (int32 tr = 0; tr < transitions_.size(); tr++) {
      transitions_[tr].pdf_id = pdf_map[transitions_[tr].pdf_id];
    }
  }


  int32 NumSequences() const { return num_sequences_; }

  // the number of PDFs (the labels on the transitions are numbered from 0 to
  // NumPdfs() - 1).
  int32 NumPdfs() const { return num_pdfs_; }
  int32 MaxNumStates() const { return max_num_hmm_states_; }
//  NumeratorGraph();

  NumeratorGraph(const Supervision &supervision, bool scale_first_transitions);

  const Int32Pair *ForwardTransitions() const;

  const Int32Pair *BackwardTransitions() const;

  const DenominatorGraphTransition *Transitions() const;

  const int32 *NumStates() const { return num_hmm_states_.data(); }

  const Matrix<BaseFloat> &FinalProbs() const { return final_probs_; }
  const Vector<BaseFloat> &FirstTransitionOffsets() const { return first_transition_offsets_; }
  bool AreFirstTransitionsScaled() const { return scale_first_transitions_; }
  void PrintInfo(bool print_transitions = false) const;

  // Use default copy constructor and assignment operator.
 private:

  void SetTransitions(const std::vector<fst::StdVectorFst> &fsts);
  /// 2-dim array of forward-transitions for a specific sequence and
  /// hmm-state: It is num_sequences_ by max_num_hmm_states_. To get the pair
  /// for seq s and state i one should use
  /// forward_transitions_.data()[s*max_num_hmm_states_ + i]
  std::vector<Int32Pair> forward_transitions_;
  std::vector<Int32Pair> backward_transitions_; //

  // This stores the actual transitions.
  std::vector<DenominatorGraphTransition> transitions_;

  /// This matrix has a size of (max_num_hmm_states_, num_sequences_) and each
  /// element gives the final prob for state i in the hmm of sequence s
  Matrix<BaseFloat> final_probs_;

  int32 num_pdfs_;
  int32 num_sequences_;
  int32 max_num_hmm_states_;
  std::vector<int32> num_hmm_states_;

  // if scale_first_transitions_ is set to true, we subtract the largest of
  // transition probabilities on arcs out of state 0, and store the offsets
  // in the following array for each sequence. This is necessary because these
  // probabilities can get too small (very large in log-scale) due to weight
  // pushing which can cause problems in numerator computations.
  bool scale_first_transitions_;
  Vector<BaseFloat> first_transition_offsets_;
};

}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_NUM_GRAPH_H_
