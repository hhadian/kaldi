// chainbin/nnet3-chain-compute-postgram.cc

// Copyright 2015  Johns Hopkins University (author: Hossein Hadian)

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
#include "hmm/transition-model.h"
#include "nnet3/nnet-chain-example.h"
#include "chain/chain-num-graph.h"
#include "chain/chain-numerator.h"
#include "chain/chain-training.h"
#include "chainbin/profiler2.h"
#include "nnet3/nnet-chain-diagnostics.h"
#include "chain/chain-denominator.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/hmm-topology.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "chain/chain-full-numerator.h"

using namespace kaldi;
using namespace kaldi::nnet3;
using namespace fst;
using namespace kaldi::chain;
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

void Compute(const NnetComputeProbOptions &nnet_config,
             const chain::ChainTrainingOptions &chain_config,
             const Nnet &nnet,
             const NnetChainExample &chain_eg,
             CuMatrix<BaseFloat>* nnet_output) {
  CachingOptimizingCompiler compiler(nnet, nnet_config.optimize_config);
  Nnet *deriv_nnet;
  if (nnet_config.compute_deriv) {
    deriv_nnet = new Nnet(nnet);
    // bool is_gradient = true;
    // SetZero(is_gradient, deriv_nnet);
  }

  bool need_model_derivative = nnet_config.compute_deriv,
       store_component_stats = false;
  ComputationRequest request;
  bool use_xent_regularization = (chain_config.xent_regularize != 0.0),
       use_xent_derivative = false;
  GetChainComputationRequest(nnet, chain_eg, need_model_derivative,
                             store_component_stats, use_xent_regularization,
                             use_xent_derivative, &request);
  const NnetComputation *computation = compiler.Compile(request);
  NnetComputer computer(nnet_config.compute_config, *computation,
                        nnet, deriv_nnet);
  // give the inputs to the computer object.
  computer.AcceptInputs(nnet, chain_eg.inputs);
  computer.Run();

  CuMatrix<BaseFloat> tmp(computer.GetOutput("output"));
  nnet_output->Resize(tmp.NumRows(), tmp.NumCols());
  nnet_output->CopyFromMat(tmp);

  //if (nnet_config.compute_deriv)
  //  computer.Run();
}


void SaveMatrixMatlab(const MatrixBase<BaseFloat> &mat, std::string fname, std::string varname) {
  std::ofstream os(fname.c_str());
  if (mat.NumCols() == 0) {
    os << varname << " = [ ];\n";
  } else {
    os << varname << " = [";
    for (MatrixIndexT i = 0; i < mat.NumRows(); i++) {
      os << "\n  ";
      for (MatrixIndexT j = 0; j < mat.NumCols(); j++)
        os << mat(i, j) << " ";
    }
    os << "\n];\n";
  }
}


void SaveMatrixSparselyMatlab(const CuMatrixBase<BaseFloat> &cumat, std::string fname, std::string varname) {
  std::ofstream os(fname.c_str());
  Matrix<BaseFloat> temp(cumat.NumRows(), cumat.NumCols(), kUndefined);
  cumat.CopyToMat(&temp);
  std::vector<int32> row_idxs, col_idxs;
  std::vector<BaseFloat> vals;

  for (MatrixIndexT i = 0; i < temp.NumRows(); i++) {
    for (MatrixIndexT j = 0; j < temp.NumCols(); j++) {
      if (temp(i, j) != 0.0) {
        row_idxs.push_back(i + 1);
        col_idxs.push_back(j + 1);
        vals.push_back(temp(i, j));
      }
    }
  }
  if (temp(temp.NumRows() - 1, temp.NumCols() - 1) == 0) {
    row_idxs.push_back(temp.NumRows());
    col_idxs.push_back(temp.NumCols());
    vals.push_back(0.0);
  }

  os << "sp_i = [ "; for (int i = 0; i < row_idxs.size(); i++) os << row_idxs[i] << " "; os << "];\n";
  os << "sp_j = [ "; for (int i = 0; i < col_idxs.size(); i++) os << col_idxs[i] << " "; os << "];\n";
  os << "sp_v = [ "; for (int i = 0; i < vals.size(); i++) os << vals[i] << " "; os << "];\n";
  os << varname << " = sparse(sp_i, sp_j, sp_v);\n";
  os << "# num-non-zero-elems: " << row_idxs.size() << "\n";
}

void SaveVectorSparselyMatlab(const Vector<BaseFloat> vec, std::string fname, std::string varname) {
  std::ofstream os(fname.c_str());
  os << varname << " = [";
  for (MatrixIndexT i = 0; i < vec.Dim(); i++) {
    os << vec(i) << " ";
  }
  os << "];\n";
}


int main(int argc, char *argv[]) {
  try {
    const char *usage =
        "Usage:  nnet3-chain-e2e-align-egs [options] <nnet3-model-in> <training-examples-in> <training-examples-out>\n";

    NnetComputeProbOptions nnet_opts;
    chain::ChainTrainingOptions chain_opts;
    std::string use_gpu = "no";
    int32 numegs = 0;
    int seed = 0;

    ParseOptions po(usage);
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("num-egs", &numegs, "");
    po.Register("srand", &seed, "");
    nnet_opts.Register(&po);
    chain_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        examples_wspecifier = po.GetArg(3);

    srand(seed);
    TransitionModel trans_model;
    AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }
    Nnet nnet = am_nnet.GetNnet();

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);
    NnetChainExampleWriter example_writer(examples_wspecifier);

    int32 num_read = 0, num_fail = 0, tot_frames = 0;
    BaseFloat totlogprob = 0, totlogprob_sqr = 0, minlogprob = 10000, maxlogprob = -10000;
    nnet_opts.compute_deriv = false;
    for (; !example_reader.Done() && (num_read < numegs || numegs == 0);
         example_reader.Next(), num_read++) {
      const NnetChainExample& eg = example_reader.Value();
      std::string key = example_reader.Key();
      KALDI_ASSERT(eg.outputs[0].supervision.num_sequences == 1);

      NumeratorGraph ng(eg.outputs[0].supervision, true);
      CuMatrix<BaseFloat> nnet_output_gpu;
      Compute(nnet_opts, chain_opts, nnet, eg, &nnet_output_gpu);
      int32 T = nnet_output_gpu.NumRows();
      int32 N = nnet_output_gpu.NumCols();
      KALDI_ASSERT(T == eg.outputs[0].supervision.frames_per_sequence);
      FullNumeratorComputation num(chain_opts, ng, nnet_output_gpu);
      BaseFloat num_logprob_weighted = num.Forward();
      KALDI_LOG << "key:" << key << ".   num logprob weighted per frame: "
                << num_logprob_weighted / T;
      if (num_logprob_weighted - num_logprob_weighted != 0) {
        KALDI_LOG << "Eg " << key << " was rejected.";
        num_fail++;
        continue;
      }
      tot_frames += T;
      totlogprob += num_logprob_weighted;
      totlogprob_sqr += num_logprob_weighted * num_logprob_weighted / T;
      if (num_logprob_weighted / T > maxlogprob)
        maxlogprob = num_logprob_weighted / T;
      if (num_logprob_weighted / T < minlogprob)
        minlogprob = num_logprob_weighted / T;
      bool ok = true;
      CuMatrix<BaseFloat> nnet_output_deriv(T, N);
      ok = num.Backward(eg.outputs[0].supervision.weight, &nnet_output_deriv);
      std::cout<< "key:" << key << "   ok: " << ok << "\n";
      if (!ok) {
        KALDI_LOG << "Eg " << key << " was rejected due to back-fail.";
        num_fail++;
        continue;
      }
      Matrix<BaseFloat> soft_ali(nnet_output_deriv);

      std::vector<std::vector<std::pair<MatrixIndexT, BaseFloat > > > rows(soft_ali.NumRows());
      for (int32 i = 0; i < soft_ali.NumRows(); i++)
        for (int32 j = 0; j < soft_ali.NumCols(); j++)
          if (soft_ali(i, j) != 0.0)
            rows[i].push_back(std::make_pair(j, soft_ali(i, j)));

      SparseMatrix<BaseFloat> ali_sparse(soft_ali.NumCols(), rows);
      //      ali_sparse.Write(std::cout, false);

      NnetChainExample eg2(eg);
      eg2.outputs[0].supervision.ali = SparseMatrix<BaseFloat>(soft_ali.NumCols(), rows);
      example_writer.Write(key, eg2);
    }
    KALDI_LOG << "Processed " << num_read << " egs. Rejected " << num_fail << " egs.";
    KALDI_LOG << "Overall logprob per frame mean: " << totlogprob / tot_frames;
    KALDI_LOG << "Overall logprob per frame stddev: " << sqrt(totlogprob_sqr / (tot_frames - 1) - (totlogprob / tot_frames) * (totlogprob / tot_frames));
    KALDI_LOG << "Overall logprob per frame min: " << minlogprob;
    KALDI_LOG << "Overall logprob per frame max: " << maxlogprob;
    return !(num_read > 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
