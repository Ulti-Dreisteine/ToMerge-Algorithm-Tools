FROM ccr.ccs.tencentyun.com/rk-ai/ai-cpu:0.0.1
WORKDIR /workspace
ENV TEMP=DEV PROJECT_REF=0.0.1 PROJECT_GROUP=ai PROJECT_NAME=srv-web-demo PROJECT_FEATURE=""
ADD . /workspace
RUN /bin/bash install.sh
CMD ["sh","./start.sh"]