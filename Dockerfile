FROM python:latest

# Env & Arg variables
ARG USERNAME=divina

# Apt update & apt install required packages
# whois: required for mkpasswd
RUN apt update && apt -y install openssh-server whois && apt install sudo

# Add a non-root user & set password
RUN useradd -ms /bin/bash $USERNAME
RUN sudo adduser $USERNAME sudo
# Save username on a file ¿?¿?¿?¿?¿?
#RUN echo "$USERNAME" > /.non-root-username

# Remove no-needed packages
RUN apt purge -y whois && apt -y autoremove && apt -y autoclean && apt -y clean
RUN apt-get update && \
    apt-get install -y curl \
    wget \
    openjdk-11-jdk

# Change to non-root user
#USER $USERNAME
#WORKDIR /home/$USERNAME

# Copy the entrypoint
COPY entrypoint.sh entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create the ssh directory and authorized_keys file 
USER $USERNAME
RUN mkdir /home/$USERNAME/.ssh && touch /home/$USERNAME/.ssh/authorized_keys
USER root

# Set volumes
VOLUME /home/$USERNAME/.ssh
VOLUME /etc/ssh

# Run entrypoint
CMD ["/entrypoint.sh"]
