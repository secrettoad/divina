FROM python:latest

# Env & Arg variables
ARG USERNAME=divina

# Apt update & apt install required packages
# whois: required for mkpasswd
RUN apt update && apt install sudo && apt install git && apt install

# Add a non-root user & set password
RUN useradd -ms /bin/bash $USERNAME
RUN sudo adduser $USERNAME sudo
# Save username on a file ¿?¿?¿?¿?¿?
#RUN echo "$USERNAME" > /.non-root-username

# Remove no-needed packages
RUN apt purge -y whois && apt -y autoremove && apt -y autoclean && apt -y clean
RUN apt-get update

# Change to non-root user
#USER $USERNAME
#WORKDIR /home/$USERNAME

# Copy the entrypoint
COPY entrypoint.sh entrypoint.sh
COPY requirements.txt requirements.txt
RUN chmod +x /entrypoint.sh

# Run entrypoint
CMD ["/entrypoint.sh"]
