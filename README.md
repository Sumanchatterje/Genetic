```mermaid
graph TD

    %% --- 1. Initiation and Order Capture (FOH/POS) ---
    A([START: Guest Seated]) --> B[1.2/1.3: Server Takes Order & Inputs Items]
    B --> C[1.4: Server Sends Order to Kitchen (Status: Pending)]

    %% --- 2. Preparation and Fulfillment (BOH/KDS) ---
    C --> D[2.1: KDS Receives Order / Cook Acknowledges Prep]
    D --> E[2.2: Cook Marks Order as Ready]
    E --> F[2.3: Server Notified & Picks Up Food]

    %% --- 3. Service, Payment Request, and Finalization ---
    F --> G[3.2: Server Requests Check / System Calculates Total]
    G --> H{3.3: Payment Successful?}

    %% Decision Branch
    H -- YES --> I[3.4: System Finalizes Transaction (Order Status: Paid)]
    H -- NO --> G

    I --> J([END: FOH Clears Table / Table Status Available])
```
