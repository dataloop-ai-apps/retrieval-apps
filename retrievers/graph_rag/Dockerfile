FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv

USER 1000
ENV HOME=/tmp
RUN pip install networkx \
    matplotlib \
    textwrap3 \
    json_repair


# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/graph-rag:0.0.11 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/graph-rag:0.0.11

# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/graph-rag:0.0.11 bash
