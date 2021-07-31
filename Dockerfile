FROM pytorch/pytorch

RUN apt-get update && apt-get install -y git gcc

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN pip install git+https://github.com/openai/CLIP.git

RUN cd /src \        
    git clone https://github.com/sergiuoprea/clip_with_few_shots.git

WORKDIR /src
