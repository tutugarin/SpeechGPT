FROM heartexlabs/label-studio:latest

RUN mkdir -p /label-studio/projects/sbs-comparison

COPY config.xml /label-studio/projects/sbs-comparison/
COPY entrypoint.sh /label-studio/projects/sbs-comparison/
COPY data/ /label-studio/projects/sbs-comparison/data/

EXPOSE 8080