"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class BaseAgent {
    constructor() {
        this.id = this.generateId();
        this.data = {};
    }
    generateId() {
        return `agent-${Math.random().toString(36).substr(2, 9)}`;
    }
    processData(input) {
        // Implement data processing logic here
        this.data = input; // Placeholder for actual processing
    }
    communicate(message) {
        // Implement communication logic here
        console.log(`Agent ${this.id} says: ${message}`);
    }
    getData() {
        return this.data;
    }
}
exports.default = BaseAgent;
