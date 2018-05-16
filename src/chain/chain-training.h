// chain/chain-training.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CHAIN_CHAIN_TRAINING_H_
#define KALDI_CHAIN_CHAIN_TRAINING_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "chain/chain-den-graph.h"
#include "chain/chain-supervision.h"

namespace kaldi {
namespace chain {


struct ChainTrainingOptions {
  // l2 regularization constant on the 'chain' output; the actual term added to
  // the objf will be -0.5 times this constant times the squared l2 norm.
  // (squared so it's additive across the dimensions).  e.g. try 0.0005.
  BaseFloat l2_regularize;

  // Coefficient for 'leaky hmm'.  This means we have an epsilon-transition from
  // each state to a special state with probability one, and then another
  // epsilon-transition from that special state to each state, with probability
  // leaky_hmm_coefficient times [initial-prob of destination state].  Imagine
  // we make two copies of each state prior to doing this, version A and version
  // B, with transition from A to B, so we don't have to consider epsilon loops-
  // or just imagine the coefficient is small enough that we can ignore the
  // epsilon loops.
  BaseFloat leaky_hmm_coefficient;


  // Cross-entropy regularization constant.  (e.g. try 0.1).  If nonzero,
  // the network is expected to have an output named 'output-xent', which
  // should have a softmax as its final nonlinearity.
  BaseFloat xent_regularize;
  bool disable_mmi;
  bool equal_align;
  bool den_use_initials, den_use_finals;
  bool viterbi;
  bool check_derivs;
  BaseFloat max_dur;
  BaseFloat num_scale;

  std::string trans_probs_filename;
  std::string write_trans_stats_prefix;
  Vector<BaseFloat> trans_probs;
  BaseFloat min_transition_prob;

  std::string pdf_map_filename;
  std::vector<int32> pdf_map;

  bool  offset_first_transitions;
  int32 prune;
  BaseFloat prune_beam;
  ChainTrainingOptions(): l2_regularize(0.0), leaky_hmm_coefficient(1.0e-05),
                          xent_regularize(0.0), disable_mmi(false),
                          equal_align(true), den_use_initials (true),
                          den_use_finals (false), viterbi(false),
                          check_derivs(false), max_dur(0), num_scale(1.0),
                          min_transition_prob(0.0),
                          offset_first_transitions(false), prune(0) { }

  void Register(OptionsItf *opts) {
    opts->Register("l2-regularize", &l2_regularize, "l2 regularization "
                   "constant for 'chain' training, applied to the output "
                   "of the neural net.");
    opts->Register("leaky-hmm-coefficient", &leaky_hmm_coefficient, "Coefficient "
                   "that allows transitions from each HMM state to each other "
                   "HMM state, to ensure gradual forgetting of context (can "
                   "improve generalization).  For numerical reasons, may not be "
                   "exactly zero.");
    opts->Register("xent-regularize", &xent_regularize, "Cross-entropy "
                   "regularization constant for 'chain' training.  If "
                   "nonzero, the network is expected to have an output "
                   "named 'output-xent', which should have a softmax as "
                   "its final nonlinearity.");
    opts->Register("disable-mmi", &disable_mmi, "Use only the num graph");
    opts->Register("equal-align", &equal_align, "Do equal align if necessary or not?");
    opts->Register("den-use-initials", &den_use_initials, "Use initial state probs in den "
                   "(all states are initial with some probability)");
    opts->Register("den-use-finals", &den_use_finals, "Use final state probs in den");
    opts->Register("viterbi", &viterbi, "Use Viterbi instead of FB");
    opts->Register("check-derivs", &check_derivs, "Double check the derivatives");
    opts->Register("max-dur", &max_dur, "Max duration of utterances in the eg (in seconds). Longer egs will not be trained on.");
    opts->Register("num-scale", &num_scale, "Bigger values  -->  more concentrated numerator occupation probs.");
    opts->Register("trans-probs-filename", &trans_probs_filename, "Double check the derivatives");
    opts->Register("write-trans-stats-prefix", &write_trans_stats_prefix, "Double check the derivatives");
    opts->Register("min-transition-prob", &min_transition_prob, "Minimum transition prob for when transition probs are used");
    opts->Register("pdf-map-filename", &pdf_map_filename, "Path to a pdf map file -- for #pdf_tying");
    opts->Register("offset-first-transitions", &offset_first_transitions, "Offset first transitions in num computation");
    opts->Register("prune", &prune, "if >=1 do pruned forward-backward");
    opts->Register("prune-beam", &prune_beam, "pruned forward-backward beam");
  }
};


/**
   This function does both the numerator and denominator parts of the 'chain'
   computation in one call.

   @param [in] opts        Struct containing options
   @param [in] den_graph   The denominator graph, derived from denominator fst.
   @param [in] supervision  The supervision object, containing the supervision
                            paths and constraints on the alignment as an FST
   @param [in] nnet_output  The output of the neural net; dimension must equal
                          ((supervision.num_sequences * supervision.frames_per_sequence) by
                            den_graph.NumPdfs()).  The rows are ordered as: all sequences
                            for frame 0; all sequences for frame 1; etc.
   @param [out] objf       The [num - den] objective function computed for this
                           example; you'll want to divide it by 'tot_weight' before
                           displaying it.
   @param [out] l2_term  The l2 regularization term in the objective function, if
                           the --l2-regularize option is used.  To be added to 'o
   @param [out] weight     The weight to normalize the objective function by;
                           equals supervision.weight * supervision.num_sequences *
                           supervision.frames_per_sequence.
   @param [out] nnet_output_deriv  The derivative of the objective function w.r.t.
                           the neural-net output.  Only written to if non-NULL.
                           You don't have to zero this before passing to this function,
                           we zero it internally.
   @param [out] xent_output_deriv  If non-NULL, then the numerator part of the derivative
                           (which equals a posterior from the numerator forward-backward,
                           scaled by the supervision weight) is written to here.  This will
                           be used in the cross-entropy regularization code.  This value
                           is also used in computing the cross-entropy objective value.
*/
void ComputeChainObjfAndDeriv(const ChainTrainingOptions &opts,
                              const DenominatorGraph &den_graph,
                              const Supervision &supervision,
                              const CuMatrixBase<BaseFloat> &nnet_output,
                              BaseFloat *objf,
                              BaseFloat *l2_term,
                              BaseFloat *weight,
                              CuMatrixBase<BaseFloat> *nnet_output_deriv,
                              CuMatrixBase<BaseFloat> *xent_output_deriv = NULL);



}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_TRAINING_H_
