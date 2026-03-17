# ANewDawn

<p align="center">
  <img src="https://github.com/thelovinator1/ANewDawn/blob/master/.github/logo.jpg?raw=true" title="A New Dawn" alt="A New Dawn" width="300" height="300" loading="lazy">
</p>

A shit Discord bot.

## Running via systemd

This repo includes a systemd unit template under `systemd/anewdawn.service` that can be used to run the bot as a service.

### Quick setup

1. Copy and edit the environment file:
   ```sh
   sudo mkdir -p /etc/ANewDawn
   sudo cp systemd/anewdawn.env.example /etc/ANewDawn/ANewDawn.env
   sudo chown -R lovinator:lovinator /etc/ANewDawn
   # Edit /etc/ANewDawn/ANewDawn.env and fill in your tokens.
   ```

2. Install the systemd unit:
   ```sh
   sudo cp systemd/anewdawn.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now anewdawn.service
   ```

3. Check status / logs:
   ```sh
   sudo systemctl status anewdawn.service
   sudo journalctl -u anewdawn.service -f
   ```
