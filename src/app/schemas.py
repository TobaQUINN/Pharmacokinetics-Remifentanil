from pydantic import BaseModel


class PKInput(BaseModel):

    Time: float
    PreviousTime: float

    Rate: float
    Amt: float
    cum_dose: float

    Age: float
    Sex: int

    Ht: float
    Wt: float

    BSA: float
    LBM: float
