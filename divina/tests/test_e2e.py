



@pytest.mark.skip('WIP')
def test_e2e_small(divina_test_version, vision_codeartifact):
    build_publish_dev.main()
    codeartifact_token = vision_codeartifact.get_authorization_token(domain='coysu')['authorizationToken']
    divina_pip_arguments = '-i https://aws:{}@coysu-169491045780.d.codeartifact.us-west-2.amazonaws.com/pypi/divina/simple/ --extra-index-url https://www.pypi.org/simple'.format(
        codeartifact_token)
    create_vision(keep_instances_alive=True, import_bucket='coysu-divina-prototype-small',
                  divina_version=divina_test_version, ec2_keyfile='divina-dev', verbosity=3,
                  divina_pip_arguments=divina_pip_arguments)
