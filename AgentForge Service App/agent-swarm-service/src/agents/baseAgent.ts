class BaseAgent {
    id: string;
    data: any;

    constructor() {
        this.id = this.generateId();
        this.data = {};
    }

    generateId(): string {
        return `agent-${Math.random().toString(36).substr(2, 9)}`;
    }

    processData(input: any): void {
        // Implement data processing logic here
        this.data = input; // Placeholder for actual processing
    }

    communicate(message: any): void {
        // Implement communication logic here
        console.log(`Agent ${this.id} says: ${message}`);
    }

    getData(): any {
        return this.data;
    }
}

export default BaseAgent;