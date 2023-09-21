const http = require('http');
const fs = require('fs');
const path = require('path');
const WebSocket = require('ws');

const server = http.createServer((req, res) => {
  const filePath = path.join(__dirname, req.url === '/' ? 'index.html' : req.url);
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('File not found');
    } else {
      res.writeHead(200);
      res.end(data);
    }
  });
});

const wss = new WebSocket.Server({ server });

const players = {};

wss.on('connection', (ws) => {
  const playerId = Math.random().toString(36).substring(7);
  players[playerId] = ws;

  ws.on('message', (message) => {
    // Broadcast the message to all players
    Object.keys(players).forEach((id) => {
      if (id !== playerId) {
        players[id].send(message);
      }
    });
  });

  ws.on('close', () => {
    delete players[playerId];
  });
});

server.listen(8080, () => {
  console.log('Server is running on http://localhost:8080');
});
