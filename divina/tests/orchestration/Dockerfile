FROM python:3.10

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

# Run entrypoint
CMD ["/entrypoint.sh"]
