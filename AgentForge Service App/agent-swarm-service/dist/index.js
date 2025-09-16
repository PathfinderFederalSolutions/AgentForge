"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const app = require("./app");
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Agent Swarm Service is running on http://localhost:${PORT}`);
});
