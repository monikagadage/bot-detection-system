# The service has no third-party runtime dependencies, so the image is just
# the code on top of a slim Python base.
FROM python:3.12-slim

WORKDIR /app
COPY . .

# Train the seed model once at build time so the first request is fast, and
# fail the build if anything is miswired.
RUN python selfcheck.py

# Bind on all interfaces inside the container; persist state to a volume.
ENV BOTSHIELD_HOST=0.0.0.0 \
    BOTSHIELD_PORT=8500 \
    BOTSHIELD_DATA=/data
VOLUME ["/data"]
EXPOSE 8500

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8500/stats').status==200 else 1)"

CMD ["python", "server.py"]
