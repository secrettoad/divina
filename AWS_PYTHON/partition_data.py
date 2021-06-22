def partition_data(vision_session, key, partition_dimensions):
    s3 = vision_session.client('s3')
    object_response = s3.get_object(
        Bucket='coysu-divina-prototype-visions/coysu-divina-prototype-{}/data'.format(
                                 os.environ['VISION_ID']),
        Key=key
    )

