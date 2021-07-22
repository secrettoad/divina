from ..vision import create_vision
from ...ops import build_publish_dev


def test_full_small(divina_test_version, vision_codeartifact):
    build_publish_dev.main()
    codeartifact_token = vision_codeartifact.get_authorization_token(domain='coysu')['authorizationToken']
    divina_pip_arguments = '-i https://aws:{}@coysu-169491045780.d.codeartifact.us-west-2.amazonaws.com/pypi/divina/simple/ --extra-index-url https://www.pypi.org/simple'.format(codeartifact_token)
    create_vision(keep_instances_alive=True, import_bucket='coysu-divina-prototype-small', divina_version=divina_test_version, ec2_keyfile='divina-dev', verbosity=3, divina_pip_arguments=divina_pip_arguments)


def test_create_infrastructure():
    pass


def test_dataset_build():
    pass


def test_train():
    pass


def test_predict():
    pass


def test_validate():
    pass
