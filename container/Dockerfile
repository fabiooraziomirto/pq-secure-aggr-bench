# TODO: select a Python base image, version and verified digest.
ARG BASE_IMAGE=TODO
FROM ${BASE_IMAGE}
WORKDIR /app
COPY . /app
# TODO: pin reviewed dependency versions and hashes, KEM backend and system packages.
RUN python -m pip install '.[test]'
CMD ["python", "-m", "pytest", "-q"]
