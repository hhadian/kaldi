// nnet3bin/nnet3-chain-train.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-chain-training.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train nnet3+chain neural network parameters with backprop and stochastic\n"
        "gradient descent.  Minibatches are to be created by nnet3-chain-merge-egs in\n"
        "the input pipeline.  This training program is single-threaded (best to\n"
        "use it with a GPU).\n"
        "\n"
        "Usage:  nnet3-chain-train [options] <raw-nnet-in> <denominator-fst-in> <chain-training-examples-in> <raw-nnet-out>\n"
        "\n"
        "nnet3-chain-train 1.raw den.fst 'ark:nnet3-merge-egs 1.cegs ark:-|' 2.raw\n";

    bool binary_write = true;
    std::string use_gpu = "yes";
    NnetChainTrainingOptions opts;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    // This relates to #pdf_tying
    if (!opts.chain_config.pdf_map_filename.empty()) {
      std::ifstream fs(opts.chain_config.pdf_map_filename);
      ReadIntegerVector(fs, false, &(opts.chain_config.pdf_map));
      KALDI_LOG << "read pdf map. size: " << opts.chain_config.pdf_map.size()
                << "\tmap[0]=" << opts.chain_config.pdf_map[0];
    }

    // This relates to transition training
    if (!opts.chain_config.trans_probs_filename.empty()) {
      ReadKaldiObject(opts.chain_config.trans_probs_filename,
                      &(opts.chain_config.trans_probs));
      for (int32 i = 0; i < opts.chain_config.trans_probs.Dim(); i += 2) {
        BaseFloat sum = (opts.chain_config.trans_probs(i) +
                         opts.chain_config.trans_probs(i + 1));
        opts.chain_config.trans_probs(i) /= sum;
        opts.chain_config.trans_probs(i + 1) /= sum;
        BaseFloat p = opts.chain_config.min_transition_prob;
        if (std::min(opts.chain_config.trans_probs(i),
                     opts.chain_config.trans_probs(i + 1))
            < p) {
          if (opts.chain_config.trans_probs(i)
              < opts.chain_config.trans_probs(i + 1)) {
            opts.chain_config.trans_probs(i) = p;
            opts.chain_config.trans_probs(i + 1) = 1.0 - p;
          } else {
            opts.chain_config.trans_probs(i) = 1.0 - p;
            opts.chain_config.trans_probs(i + 1) = p;
          }
        }
      }
      KALDI_LOG << "Transitions probs:";
      opts.chain_config.trans_probs.Write(std::cerr, false);
      opts.chain_config.trans_probs.ApplyLog();
    }


#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
        den_fst_rxfilename = po.GetArg(2),
        examples_rspecifier = po.GetArg(3),
        nnet_wxfilename = po.GetArg(4);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    bool ok;

    {
      fst::StdVectorFst den_fst;
      ReadFstKaldi(den_fst_rxfilename, &den_fst);

      NnetChainTrainer trainer(opts, den_fst, &nnet);

      SequentialNnetChainExampleReader example_reader(examples_rspecifier);
      int32 tot_frames = 0, discarded_frames = 0;
      for (; !example_reader.Done(); example_reader.Next()) {
        const NnetChainExample &eg = example_reader.Value();
        std::cerr << "\n";
        KALDI_LOG << "Training on minibatch " << example_reader.Key()
                  << "  T: " << eg.outputs[0].supervision.frames_per_sequence
                  << "  N: " << eg.outputs[0].supervision.num_sequences;
        tot_frames += eg.outputs[0].supervision.frames_per_sequence *
            eg.outputs[0].supervision.num_sequences;
        int32 max_frames = static_cast<int32>(opts.chain_config.max_dur * 100.0 / 3.0);
        if (opts.chain_config.max_dur > 0 &&
            eg.outputs[0].supervision.frames_per_sequence > max_frames) {
          KALDI_LOG << "Minibatch discarded because it is too long: "
                    << "max-frames = " << max_frames;
          discarded_frames += eg.outputs[0].supervision.frames_per_sequence *
              eg.outputs[0].supervision.num_sequences;
          continue;
        }
        trainer.Train(eg);
      }

      ok = trainer.PrintTotalStats();
      KALDI_LOG << "Total number of frames: "
                << tot_frames << " discarded: "
                << discarded_frames << " (" << 100.0 * discarded_frames / tot_frames
                << " percent).";
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
    KALDI_LOG << "Wrote raw model to " << nnet_wxfilename;
    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
