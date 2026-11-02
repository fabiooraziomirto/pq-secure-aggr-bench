# Crypto/agent image (CPU only): sites, reducer and controller use the same image.
# Not built in the authoring session. Pin the base by digest at build time and
# record it in docs/decisions.md:
#   docker build -f container/Dockerfile \
#     --build-arg BASE_IMAGE=ubuntu:22.04@sha256:<resolved-digest> -t pqfl-crypto .
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS oqs-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libssl-dev && rm -rf /var/lib/apt/lists/*
# liboqs 0.16.0, the commit verified in docs/decisions.md; the build fails on mismatch.
ARG LIBOQS_COMMIT=5a1a854b0dc9f2141bdc771c555ee60c37950183
RUN git clone --depth 1 --branch 0.16.0 https://github.com/open-quantum-safe/liboqs.git /src/liboqs \
 && test "$(git -C /src/liboqs rev-parse HEAD)" = "${LIBOQS_COMMIT}" \
 && cmake -S /src/liboqs -B /build -DCMAKE_INSTALL_PREFIX=/opt/oqs -DBUILD_SHARED_LIBS=ON \
        -DOQS_BUILD_ONLY_LIB=ON -DOQS_ENABLE_KEM_ml_kem_768=ON -DOQS_ENABLE_SIG_ml_dsa_65=ON \
 && cmake --build /build --parallel 4 && cmake --install /build

FROM ${BASE_IMAGE}
# Ubuntu 22.04 matches the platform of the official OpenFHE wheel (Python 3.10).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip iproute2 libssl3 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=oqs-build /opt/oqs /opt/oqs
ENV OQS_INSTALL_PATH=/opt/oqs LD_LIBRARY_PATH=/opt/oqs/lib PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY configs /app/configs
COPY container/entrypoint.sh /usr/local/bin/entrypoint.sh
# Extras pin liboqs-python==0.16.0.1 and openfhe==1.6.1.0.22.4. A hash-locked
# requirements file is still TODO; the resolved set is frozen into the image.
RUN python3 -m venv /opt/venv && pip install --no-cache-dir '.[real,he]' \
 && pip freeze > /opt/pqfl-freeze.txt && chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
