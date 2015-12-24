// durmodbin/durmod-make-egs.cc
// Author: Hossein Hadian

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
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "tree/build-tree.h"
#include "util/parse-options.h"
#include "durmod/kaldi-durmod.h"
int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  using nnet3::NnetExample;

  try {
    const char *usage =
        "Prepare training examples for the phone duration model.\n"
        "Usage:  durmod-make-egs [options] <dur-model> <trans-model> "
        "<alignments-rspecifier> <egs-wspecifier>\n"
        "e.g.: \n"
        "  durmod-make-egs 0.durmod final.mdl egs:ali.1 ark:1.egs";
    ParseOptions po(usage);
    po.Read(argc, argv);
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    std::string durmodel_filename = po.GetArg(1),
                transmodel_filename = po.GetArg(2),
                alignments_rspecifier = po.GetArg(3),
                examples_wspecifier = po.GetArg(4);
    TransitionModel trans_model;
    ReadKaldiObject(transmodel_filename, &trans_model);
    PhoneDurationModel durmodel;
    ReadKaldiObject(durmodel_filename, &durmodel);
    SequentialInt32VectorReader reader(alignments_rspecifier);
    TableWriter<KaldiObjectHolder<NnetExample> >
                                           example_writer(examples_wspecifier);
    int32 n_done = 0;
    int32 n_egs_done = 0;
    PhoneDurationEgsMaker egs_maker(durmodel);
    KALDI_LOG << "Feature dim: " << egs_maker.FeatureDim()
              << ", NumBinaryFeatures: " << egs_maker.NumBinaryFeatures()
              << ", NumPhoneIdentities: " << egs_maker.NumPhoneIdentities();
    // Temporary statistics -- for debugging
    BaseFloat mean_duration = 0, max_duration = -1, min_duration = 1000;
    BaseFloat lognormal_mean = 0, lognormal_var = 0;
    std::vector<std::pair<int32, int32> > tmp_allpairs;
    for (; !reader.Done(); reader.Next()) {
      std::string key = reader.Key();
      const std::vector<int32> &alignment = reader.Value();
      KALDI_LOG << "Alignment Key: " << key << ", Size: " << alignment.size();
      std::vector<std::vector<int32> > split;
      SplitToPhones(trans_model, alignment, &split);
      std::vector<std::pair<int32, int32> > pairs;
      for (size_t i = 0; i < split.size(); i++) {
        KALDI_ASSERT(split[i].size() > 0);
        int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
        int32 num_repeats = split[i].size();
        KALDI_ASSERT(num_repeats != 0);
        pairs.push_back(std::make_pair(phone, num_repeats));
        tmp_allpairs.push_back(std::make_pair(phone, num_repeats));
        if (num_repeats > max_duration)
          max_duration = num_repeats;
        if (num_repeats < min_duration)
          min_duration = num_repeats;
        mean_duration += num_repeats;
        lognormal_mean += Log(static_cast<float>(num_repeats));
      }
      std::vector<NnetExample> egs;
      egs_maker.AlignmentToNnetExamples(pairs, &egs);
      n_egs_done += egs.size();
      for (int i = 0; i < egs.size(); i++)
        example_writer.Write(key, egs[i]);
      n_done++;
    }
    mean_duration /= n_egs_done;
    lognormal_mean /= n_egs_done;
    for (int i = 0; i < tmp_allpairs.size(); i++) {
      BaseFloat duration_in_frames = static_cast<float>(tmp_allpairs[i].second);
      BaseFloat squared_logduration = Log(duration_in_frames) - lognormal_mean;
      lognormal_var += squared_logduration * squared_logduration;
    }
    lognormal_var /= n_egs_done;
    KALDI_LOG << "Wrote " << n_egs_done << " examples.";
    // My estimation of a reasonable max_duration_ using stats:
    int32 stats_based_max_duration_ = exp(lognormal_mean)
                                      + 2.5 * exp(sqrt(lognormal_var));

    KALDI_LOG << "Statistics of durations (in frames): "
              << " Mean=" << mean_duration
              << " Min=" << min_duration
              << " Max=" << max_duration
              << " LogNormal-Mean=" << lognormal_mean
              << " LogNormal-SD=" << sqrt(lognormal_var)
              << " Stats-based-max-duration: " << stats_based_max_duration_;
    KALDI_LOG << "Done " << n_done << " utterances.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
