export class OutputService {
    formatResult(result: any): string {
        // Implement formatting logic here
        return JSON.stringify(result, null, 2);
    }

    deliverResult(formattedResult: string, destination: string): void {
        // Implement delivery logic here
        console.log(`Delivering result to ${destination}: ${formattedResult}`);
    }
}