#!/usr/bin/env bash
exec /usr/bin/ssh -p 2222 -i /home/votrongnam/.ssh/id_rsa -o StrictHostKeyChecking=accept-new "$@"
