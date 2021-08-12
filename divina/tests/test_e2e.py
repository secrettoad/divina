



@pytest.mark.skip('WIP')
def test_e2e_small(divina_test_version, vision_codeartifact):

    build_publish_dev.main()
    create_vision(keep_instances_alive=True, import_bucket='coysu-divina-prototype-small',
                  divina_version=divina_test_version, ec2_keyfile='divina-dev', verbosity=3,
                  divina_pip_arguments=divina_pip_arguments)
