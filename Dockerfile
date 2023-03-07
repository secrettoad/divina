FROM python:3.10.0

COPY requirements.txt /divina/requirements.txt

RUN pip install -r divina/requirements.txt

ARG buildtime_variable='vtest.0.0.1'
ENV PBR_VERSION=$buildtime_variable

COPY . /divina

RUN python3 -m pip install -e divina

COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

# Run entrypoint
CMD ["/entrypoint.sh"]
