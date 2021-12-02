FROM python:3.7

# Env & Arg variables
ARG USERNAME=divina
ARG USERPASS=divina

# Apt update & apt install required packages
# whois: required for mkpasswd
RUN apt update && apt -y install openssh-server whois && apt install sudo && apt install git && apt install

# Add a non-root user & set password
RUN useradd -ms /bin/bash $USERNAME
RUN sudo adduser $USERNAME sudo
RUN usermod --password $(echo "$USERPASS" | mkpasswd -s) $USERNAME

# Save username on a file ¿?¿?¿?¿?¿?
#RUN echo "$USERNAME" > /.non-root-username

# Remove no-needed packages
RUN apt purge -y whois && apt -y autoremove && apt -y autoclean && apt -y clean
RUN apt-get update

# Change to non-root user
#USER $USERNAME
#WORKDIR /home/$USERNAME



COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt


# Create the ssh directory and authorized_keys file 
USER $USERNAME
RUN mkdir /home/$USERNAME/.ssh && touch /home/$USERNAME/.ssh/authorized_keys
USER root

# Set volumes
VOLUME /home/$USERNAME/.ssh
VOLUME /etc/ssh

# Run entrypoint
CMD ["/entrypoint.sh"]
