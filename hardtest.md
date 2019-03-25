# Hard Test for XGBoost
## Code for my:loss inserted in my_obj.cc
```
#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <utility>

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(my_obj);

class MyLossObj : public ObjFunction {
public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {}

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                  const MetaInfo& info,
                  int iter,
                  HostDeviceVector<GradientPair>* out_gpair) override {
    /* Boilerplate */
    CHECK_EQ(preds.Size(), info.labels_.Size());
    const auto& yhat = preds.HostVector();
    const auto& y = info.labels_.HostVector();
    out_gpair->Resize(y.size());
    std::vector<GradientPair>& gpair = out_gpair->HostVector();

    const omp_ulong ndata = static_cast<omp_ulong>(yhat.size()); // NOLINT(*)
    // Implementation for your loss function goes here
    // TODO: Read from yhat (predicted labels) and y (true labels) and
    // assign first/second-order gradients to gpair, as follows:
    //
    //   gpair[i] = GradientPair( [first-order grad], [second-order grad] )
    // ...
    double grad, hess;
    for(double i = 0; i < ndata; i++)
    {
      grad = (yhat[i] > y[i])?((0.5)*(yhat[i] - y[i])):((-2)*(y[i] - yhat[i]));
      hess = (yhat[i] > y[i])?0.5:2;
      //if(yhat[i] > y[i])
        //grad = (0.5)*(yhat[i] - y[i]);
      //else grad = (-2)*(y[i] - yhat[i]);
      gpair.at(i) = GradientPair(grad, hess);
    }
    
  }
  const char* DefaultEvalMetric() const override {
    return "rmse";
  }
};

// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(MyLossObj, "my:loss")
.describe("My very first loss function")
.set_body([]() { return new MyLossObj(); });

}  // namespace obj
}  // namespace xgboost
```
## Testing my:loss in agaricus dataset
```
rm(list = ls())
require(xgboost)
library(Matrix)
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)
watchlist <- list(train = dtrain, eval = dtest)

## A simple xgb.train example:
param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2,
              objective = "my:loss")
bst <- xgb.train(param, dtrain, nrounds = 6, watchlist)
```
## Console Output
```
> bst <- xgb.train(param, dtrain, nrounds = 6, watchlist)
[1]	train-rmse:0.230485	eval-rmse:0.235568 
[2]	train-rmse:0.071054	eval-rmse:0.056878 
[3]	train-rmse:0.066753	eval-rmse:0.057403 
[4]	train-rmse:0.051132	eval-rmse:0.052407 
[5]	train-rmse:0.031440	eval-rmse:0.016874 
[6]	train-rmse:0.030281	eval-rmse:0.018359 
```
