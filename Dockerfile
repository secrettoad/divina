FROM python:3.10

COPY . /divina

RUN python3 -m pip install -e divina

COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

# Run entrypoint
CMD ["/entrypoint.sh"]
