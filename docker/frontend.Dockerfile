# A static site, deliberately no Node/npm build step -- vanilla HTML/CSS/JS
# (frontend/app.js) is enough for an internal approval-review dashboard, and
# skipping a JS toolchain keeps this container (and its rebuild time) small.
FROM nginx:1.29-alpine

COPY frontend/ /usr/share/nginx/html/
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
