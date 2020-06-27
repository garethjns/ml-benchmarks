from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier

from ml_benchmarks.models.base_model import BaseModel


class XGBoostSKGrid(BaseModel):
    def __init__(self, *args, verbose: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose

    def build(self):
        grid = {'n_estimators': [50, 100, 150, 200],
                'max_depth': [2, 4, 6],
                'colsample_bytree': [0.5, 0.75, 1]}

        gs = GridSearchCV(XGBClassifier(n_jobs=self.n_jobs),
                          param_grid=grid,
                          n_jobs=self.n_jobs,
                          verbose=self.verbose)

        return gs
