FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv

RUN ${DL_PYTHON_EXECUTABLE} -m pip install networkx \
    matplotlib \
    textwrap3 \
    json_repair


# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/graph-rag:0.0.12 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/graph-rag:0.0.12

# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/graph-rag:0.0.12 bash
