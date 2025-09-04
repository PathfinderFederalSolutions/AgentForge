export type Agent = {
    id: string;
    name: string;
    status: 'active' | 'inactive' | 'error';
    processData: (data: any) => Promise<any>;
};

export type Coordinator = {
    agents: Agent[];
    assignTask: (agentId: string, task: any) => void;
    getStatus: () => Record<string, string>;
};

export type Workflow = {
    id: string;
    steps: Array<() => Promise<any>>;
    execute: () => Promise<any>;
};

export type IngestionService = {
    source: string;
    fetchData: () => Promise<any>;
};

export type ProcessingService = {
    process: (data: any) => Promise<any>;
};

export type OutputService = {
    format: (data: any) => string;
    deliver: (formattedData: string) => Promise<void>;
};