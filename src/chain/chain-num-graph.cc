// chain/chain-num-graph.cc

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


#include "chain/chain-num-graph.h"
#include "hmm/hmm-utils.h"
#include "fstext/push-special.h"

namespace kaldi {
namespace chain {


NumeratorGraph::NumeratorGraph(const Supervision &supervision,
                               bool scale_first_transitions) {
  scale_first_transitions_ = scale_first_transitions;
  num_sequences_ = supervision.num_sequences;
  num_pdfs_ = supervision.label_dim;
  max_num_hmm_states_ = 0;
  num_hmm_states_.resize(num_sequences_);
  KALDI_ASSERT(supervision.e2e_fsts.size() == num_sequences_);

  for (int32 i = 0; i < num_sequences_; i++) {
    KALDI_ASSERT(supervision.e2e_fsts[i].Properties(fst::kIEpsilons, true) == 0);
    num_hmm_states_[i] = supervision.e2e_fsts[i].NumStates();
    if (num_hmm_states_[i] > max_num_hmm_states_)
      max_num_hmm_states_ = num_hmm_states_[i];
  }
  final_probs_.Resize(max_num_hmm_states_, num_sequences_, kSetZero);
  SetTransitions(supervision.e2e_fsts);
  // SetFinalProbs();
}

const Int32Pair* NumeratorGraph::BackwardTransitions() const {
  return backward_transitions_.data();
}

const Int32Pair* NumeratorGraph::ForwardTransitions() const {
  return forward_transitions_.data();
}

const DenominatorGraphTransition* NumeratorGraph::Transitions() const {
  return transitions_.data();
}

//const CuMatrix<BaseFloat>& NumeratorGraph::FinalProbs() const {
//  return final_probs_;
//}

void NumeratorGraph::SetTransitions(
                                   const std::vector<fst::StdVectorFst> &e2e_fsts) {

  int32 transitions_dim = num_sequences_ * max_num_hmm_states_;

  std::vector<std::vector<DenominatorGraphTransition> >
      transitions_out(transitions_dim),
      transitions_in(transitions_dim);
  std::vector<Int32Pair> forward_transitions(transitions_dim);
  std::vector<Int32Pair> backward_transitions(transitions_dim);
  std::vector<DenominatorGraphTransition> transitions;
  Vector<BaseFloat> offsets(num_sequences_);
  for (int32 seq = 0; seq < num_sequences_; seq++) {
    for (int32 s = 0; s < e2e_fsts[seq].NumStates(); s++) {
      final_probs_(s, seq) = exp(-e2e_fsts[seq].Final(s).Value());
      BaseFloat offset = 0.0;
      if (s == 0 && scale_first_transitions_) {
        for (fst::ArcIterator<fst::StdVectorFst> aiter(e2e_fsts[seq], s);
             !aiter.Done();
             aiter.Next())
          if (aiter.Value().weight.Value() > offset)
            offset = aiter.Value().weight.Value();
        offsets(seq) = -offset;
      }
      for (fst::ArcIterator<fst::StdVectorFst> aiter(e2e_fsts[seq], s);
           !aiter.Done();
           aiter.Next()) {
        const fst::StdArc &arc = aiter.Value();
        DenominatorGraphTransition transition;
        transition.transition_prob = exp(-(arc.weight.Value() - offset));
        transition.pdf_id = arc.ilabel - 1;
        transition.hmm_state = arc.nextstate;  // it is local (i.e. within
                                               // the corresponding hmm)
        KALDI_ASSERT(transition.pdf_id >= 0 && transition.pdf_id < num_pdfs_);
        transitions_out[seq * max_num_hmm_states_ + s].push_back(transition);
        // now the reverse transition.
        transition.hmm_state = s;  // it is local (i.e. within the
                                   // corresponding hmm)
        transitions_in[seq * max_num_hmm_states_ + arc.nextstate].push_back(
                                                                    transition);
      }
    }
  }
  first_transition_offsets_ = offsets;
  for (int32 s = 0; s < transitions_dim; s++) {
    forward_transitions[s].first = static_cast<int32>(transitions.size());
    transitions.insert(transitions.end(), transitions_out[s].begin(),
                       transitions_out[s].end());
    forward_transitions[s].second = static_cast<int32>(transitions.size());
  }
  for (int32 s = 0; s < transitions_dim; s++) {
    backward_transitions[s].first = static_cast<int32>(transitions.size());
    transitions.insert(transitions.end(), transitions_in[s].begin(),
                       transitions_in[s].end());
    backward_transitions[s].second = static_cast<int32>(transitions.size());
  }

  forward_transitions_ = forward_transitions;
  backward_transitions_ = backward_transitions;
  transitions_ = transitions;
}

void NumeratorGraph::PrintInfo(bool print_transitions) const {
  std::cout << "NumPdfs: " << NumPdfs() << "\n"
            << "NumSequences: " << NumSequences() << "\n"
            << "forward-transitions dim: " << forward_transitions_.size() << "\n"
            << "backward-transitions dim: " << backward_transitions_.size() << "\n"
            << "transitions dim: " << transitions_.size() << "\n"
            << "Approximate Mem Usage: " <<
            ((forward_transitions_.size() + backward_transitions_.size()) *
            sizeof(Int32Pair) + transitions_.size() *
            sizeof(DenominatorGraphTransition)) / 1024 / 1024 << " MBytes.\n"
            ;
  if (!print_transitions)
    return;
  for (int seq = 0; seq < NumSequences(); seq++) {
    std::cout << "\n\n------ SEQUENCE " << seq << " ------\n"
              << "num-states: " << NumStates()[seq] << "\n";
    std::cout << "FORWARD TRANSITIONS:\n";
    for (int i = 0; i < NumStates()[seq]; i++) {
      int from = i;
      for (int j = ForwardTransitions()[seq * MaxNumStates() + i].first;
           j < ForwardTransitions()[seq * MaxNumStates() + i].second; j++) {
        int to = Transitions()[j].hmm_state;
        BaseFloat weight = Transitions()[j].transition_prob;
        std::cout << "(" << from << " -> " << to << "): " << weight << "\n";
      }
    }
  }
}


}  // namespace chain
}  // namespace kaldi
