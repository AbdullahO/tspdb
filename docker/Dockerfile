FROM postgres:14
RUN apt-get update
RUN apt-get -y install postgresql-plpython3-14
RUN apt-get -y install git
RUN apt-get install -y curl
RUN apt-get install python3-distutils -y
RUN git clone https://github.com/AbdullahO/tspdb
RUN rm /usr/lib/python3.11/EXTERNALLY-MANAGED
RUN cd /usr/lib/python3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN pip3 install /tspdb/
RUN cp /tspdb/extension_py3/tspdb.control /usr/share/postgresql/14/extension/tspdb.control
RUN cp /tspdb/extension_py3/tspdb--0.0.1.sql /usr/share/postgresql/14/extension/tspdb--0.0.1.sql
