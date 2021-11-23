from .train import _train
from .forecast import _forecast
from .validate import _validate
from .utils import validate_experiment_definition


@validate_experiment_definition
def _experiment(experiment_definition, read_path, write_path, random_state=None, s3_fs=None):
    _train(
        s3_fs=s3_fs,
        experiment_definition=experiment_definition,
        write_path=write_path,
        random_seed=random_state
    )
    _forecast(
        s3_fs=s3_fs,
        experiment_definition=experiment_definition,
        read_path=read_path,
        write_path=write_path
    )
    _validate(
        s3_fs=s3_fs,
        experiment_definition=experiment_definition,
        read_path=read_path,
        write_path=write_path
    )
