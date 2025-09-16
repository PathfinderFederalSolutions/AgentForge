"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.logger = exports.logDebug = exports.logError = exports.logInfo = void 0;
const winston_1 = require("winston");
const logger = (0, winston_1.createLogger)({
    level: 'info',
    format: winston_1.format.combine(winston_1.format.timestamp(), winston_1.format.printf((info) => {
        // info: TransformableInfo, always has timestamp, level, message
        return `${info.timestamp} [${info.level}]: ${info.message}`;
    })),
    transports: [
        new winston_1.transports.Console(),
        new winston_1.transports.File({ filename: 'logs/error.log', level: 'error' }),
        new winston_1.transports.File({ filename: 'logs/combined.log' })
    ],
});
exports.logger = logger;
const logInfo = (message) => {
    logger.info(message);
};
exports.logInfo = logInfo;
const logError = (message) => {
    logger.error(message);
};
exports.logError = logError;
const logDebug = (message) => {
    logger.debug(message);
};
exports.logDebug = logDebug;
