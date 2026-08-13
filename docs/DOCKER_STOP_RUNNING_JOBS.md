# Stop Docker Jobs

Khi Docker run cua project van dang chay va ton RAM/CPU:

```bash
docker ps --format "{{.Names}}" | xargs -r docker stop
```

Lenh nay dung tat ca container Docker dang chay.

Kiem tra truoc/sau khi dung:

```bash
docker ps
docker stats
free -h
```

Neu `docker stop` khong dung duoc, ep kill:

```bash
docker ps --format "{{.Names}}" | xargs -r docker kill
```

Neu job dang chay ngay trong terminal hien tai thi bam:

```text
Ctrl+C
```
