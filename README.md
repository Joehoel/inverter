# Inverter

The goal of this project is to have an easy way to invert PowerPoints with Images from black on white to white on black. I could't find an easy way to this other than in Python with `python-pptx` and `pillow`
Thats why I initially created a version of this application in `Streamlit`. This was very easy to make but it does not allow for much control / customization. In the future I want to add a Raycast extension for this and maybe more things that can interface with my API.

Previous application: https://powerpoint-inverter.streamlit.app/

## Stack

- SST
  - Lambda
    - Python
  - API Gateway
  - S3
  - Cron (not yet)
- Remix

## TODO

- [ ] Add CI/CD
- [ ] Improve frontend UI.
- [ ] Create Raycast extension

---

Made with ❤️ by Joël Kuijper
