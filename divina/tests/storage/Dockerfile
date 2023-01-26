FROM python:3.10

RUN wget https://dl.min.io/server/minio/release/linux-amd64/archive/minio_20221202191922.0.0_amd64.deb -O minio.deb
RUN dpkg -i minio.deb
RUN mkdir ~/minio

COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

# Run entrypoint
CMD ["/entrypoint.sh"]
