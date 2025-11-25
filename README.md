```mermaid
flowchart LR

    %% Actors
    A[\"Customer\"]
    B[\"Server / FOH Staff\"]
    C[\"Kitchen Staff\"]
    D[\"Restaurant Manager\"]
    E[\"Inventory Staff\"]


    %% Use Cases
    UC1((Place Order))
    UC2((Make Payment))
    UC3((View Menu))
    UC4((Serve Food))
    UC5((Prepare Order))
    UC6((Mark Order Ready))
    UC7((Check Reservations))
    UC8((Manage Waitlist))
    UC9((Generate Sales Report))
    UC10((Update Inventory))
    UC11((Check Stock Levels))

    %% Relationships
    A --> UC1
    A --> UC2
    A --> UC3

    B --> UC1
    B --> UC4
    B --> UC7
    B --> UC8

    C --> UC5
    C --> UC6

    D --> UC9

    E --> UC10
    E --> UC11
```
