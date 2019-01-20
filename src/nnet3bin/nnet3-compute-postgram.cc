// nnet3bin/nnet3-compute-postgram.cc

// Copyright 2012-2015   Johns Hopkins University (author: Daniel Povey)
//                2017   Hossein Hadian

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
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"


using namespace kaldi;
using namespace kaldi::nnet3;
using namespace fst;
typedef kaldi::int32 int32;
typedef kaldi::int64 int64;

void ReadSharedPhonesList(std::string rxfilename, std::vector<std::vector<int32> > *list_out) {
  list_out->clear();
  Input input(rxfilename);
  std::istream &is = input.Stream();
  std::string line;
  while (std::getline(is, line)) {
    list_out->push_back(std::vector<int32>());
    if (!SplitStringToIntegers(line, " \t\r", true, &(list_out->back())))
      KALDI_ERR << "Bad line in shared phones list: " << line << " (reading "
                << PrintableRxfilename(rxfilename) << ")";
    std::sort(list_out->rbegin()->begin(), list_out->rbegin()->end());
    if (!IsSortedAndUniq(*(list_out->rbegin())))
      KALDI_ERR << "Bad line in shared phones list (repeated phone): " << line
                << " (reading " << PrintableRxfilename(rxfilename) << ")";
  }
}

int main(int argc, char *argv[]) {
  try {
    const char *usage =
        "Propagate the features through raw neural network model "
        "and write the output with the requested granularity.\n"
        "If --apply-exp=true, apply the Exp() function to the output "
        "before writing it out.\n"
        "\n"
        "Usage: nnet3-compute [options] <nnet-am-in> <features-rspecifier> <matrix-wspecifier>\n"
        " e.g.: nnet3-compute final.mdl scp:feats.scp ark:nnet_prediction.ark\n";

    ParseOptions po(usage);

    NnetSimpleComputationOptions opts;
    opts.acoustic_scale = 1.0; // by default do no scaling in this recipe.
    int32 num_utts = 1;
    std::string granularity = "sets";
    std::string phone_sets_file = "";
    bool apply_exp = true;
    std::string use_gpu = "no";


    std::string ivector_rspecifier,
                online_ivector_rspecifier,
                utt2spk_rspecifier;
    int32 online_ivector_period = 0;
    opts.Register(&po);


    po.Register("granularity", &granularity, "sets|phones|pdf");
    po.Register("phone-sets-file", &phone_sets_file,
                "Phone sets file. Required if granularity=sets");
    po.Register("num-utts", &num_utts, "How many of the input utterances to process. "
                "Exits after processing num-utts utterances");
    po.Register("ivectors", &ivector_rspecifier, "Rspecifier for "
                "iVectors as vectors (i.e. not estimated online); per utterance "
                "by default, or per speaker if you provide the --utt2spk option.");
    po.Register("utt2spk", &utt2spk_rspecifier, "Rspecifier for "
                "utt2spk option used to get ivectors per speaker");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier for "
                "iVectors estimated online, as matrices.  If you supply this,"
                " you must set the --online-ivector-period option.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of frames "
                "between iVectors in matrices supplied to the --online-ivectors "
                "option");
    po.Register("apply-exp", &apply_exp, "If true, apply exp function to "
                "output");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif


    std::string nnet_rxfilename = po.GetArg(1),
                feature_rspecifier = po.GetArg(2),
                matrix_wspecifier = po.GetArg(3),
                score_wspecifier = po.GetOptArg(4),
                stat_wspecifier = po.GetOptArg(5);

    TransitionModel trans_model;
    AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }
    Nnet& nnet = am_nnet.GetNnet();
    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);

    std::map<int, int> phone_map;
    std::map<int, int> pid_map;  // maps PdfId's to phone-ids (phone-ids can be
                                 // either actual phones (phones.txt) or members
                                 // of a reduced set of phones (e.g. sets.txt))
    int32 granularity_dim = trans_model.NumPdfs();
    if (granularity == "sets") {
      KALDI_ASSERT(phone_sets_file != "");
      std::vector<std::vector<int32> > shared_phones;
      ReadSharedPhonesList(phone_sets_file, &shared_phones);
      granularity_dim = shared_phones.size();
      for (int i = 0; i < shared_phones.size(); i++)
        for (int j = 0; j < shared_phones[i].size(); j++)
          phone_map[shared_phones[i][j]] = i;
    } else if (granularity == "phones") {  // This is really not efficient
      const std::vector<int32> phones = trans_model.GetPhones();
      granularity_dim = phones.size();
      for (int i = 0; i < phones.size(); i++)
        phone_map[i] = i;
    }
    if (granularity == "sets" || granularity == "phones") {
      int32 num_tids = trans_model.NumTransitionIds();
      for (int tid = 1; tid <= num_tids; tid++) {
        int32 pdfid = trans_model.TransitionIdToPdf(tid);
        int32 phone = trans_model.TransitionIdToPhone(tid);
        pid_map[pdfid] = phone_map[phone];   // TODO(hhadian)  ---> this won't be right for tied states as they can belong to multiple phones
      }
    }

    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);
    RandomAccessBaseFloatVectorReaderMapped ivector_reader(
        ivector_rspecifier, utt2spk_rspecifier);
    CachingOptimizingCompiler compiler(nnet, opts.optimize_config);
    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    BaseFloatVectorWriter score_writer(score_wspecifier);
    std::ofstream stat_outstream;
    if (stat_wspecifier != "")
      stat_outstream.open(stat_wspecifier);

    int32 num_success = 0, num_fail = 0;
    int64 frame_count = 0;
    int32 num_read = 0;
    for (; !feature_reader.Done() && num_read < num_utts;
           feature_reader.Next(), num_read++) {
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &features (feature_reader.Value());
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }
      const Matrix<BaseFloat> *online_ivectors = NULL;
      const Vector<BaseFloat> *ivector = NULL;
      if (!ivector_rspecifier.empty()) {
        if (!ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No iVector available for utterance " << utt;
          num_fail++;
          continue;
        } else {
          ivector = &ivector_reader.Value(utt);
        }
      }
      if (!online_ivector_rspecifier.empty()) {
        if (!online_ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No online iVector available for utterance " << utt;
          num_fail++;
          continue;
        } else {
          online_ivectors = &online_ivector_reader.Value(utt);
        }
      }

      Vector<BaseFloat> priors;
      DecodableNnetSimple nnet_computer(
          opts, nnet, priors,
          features, &compiler,
          ivector, online_ivectors,
          online_ivector_period);
      int32 T = nnet_computer.NumFrames(),
            N = nnet_computer.OutputDim();
      Matrix<BaseFloat> nnet_output(T, N);
      for (int32 t = 0; t < T; t++) {
        SubVector<BaseFloat> row(nnet_output, t);
        nnet_computer.GetOutputForFrame(t, &row);
      }
      nnet_output.ApplyExp();


      Matrix<BaseFloat> out(T, granularity_dim, kSetZero);
      Vector<BaseFloat> scores;
      if (score_wspecifier != "")
        scores.Resize(T, kSetZero);
      if (granularity == "pdf") {
        out = nnet_output;
      } else {
        for (int32 t = 0; t < T; t++) {
          BaseFloat sum = 0;
          for (int32 n = 0; n < N; n++) {
            out(t, pid_map[n]) += nnet_output(t, n);
            sum += nnet_output(t, n);
          }
          for (int32 i = 0; i < out.NumCols(); i++)
            out(t, i) /= sum;  // has no effect for nnet3/CE and end2end chain
        }
      }

      if (!apply_exp)
        out.ApplyLog();
      matrix_writer.Write(utt, out);

      if (scores.Dim() != 0) {
        for (int32 t = 0; t < T; t++) {
          BaseFloat entropy = 0;
          for (int32 i = 0; i < out.NumCols(); i++)
            if (out(t, i) != 0.0)
              entropy -= Log(out(t, i)) * out(t, i);
          BaseFloat max_entropy = Log(1.0 * out.NumCols());
          BaseFloat score = (1.0 - entropy / max_entropy) * 100;
          scores(t) = (int32)score;
        }
        score_writer.Write(utt, scores);

        if (stat_wspecifier != "") {
          BaseFloat mean = scores.Sum() / T;
          BaseFloat norm2 = scores.Norm(2.0);
          BaseFloat stdev = std::sqrt(norm2 * norm2 / T - mean * mean);
          stat_outstream << utt << "  mean: " << mean
                         << "  std:" << stdev
                         << "  min:" << scores.Min()
                         << "  max:" << scores.Max()
                         << "\n";
        }
      }

      frame_count += features.NumRows();
      num_success++;
    }

    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
