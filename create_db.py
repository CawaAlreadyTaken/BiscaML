import random
from typing import Any
import pandas as pd

FROM_NUMBER_TO_VALUE = [
    "Asso",
    "Due",
    "Tre",
    "Quattro",
    "Cinque",
    "Sei",
    "Sette",
    "Fante",
    "Cavallo",
    "Re"
]

FROM_NUMBER_TO_SEGNO = [
    "Bastoni",
    "Spade",
    "Coppe",
    "Denari" # "Ori" :)
]

class Carta:
    def __init__(self, id: int):
        self.id = id
        self.valore = FROM_NUMBER_TO_VALUE[self.id % 10]
        self.segno = FROM_NUMBER_TO_SEGNO[self.id // 10]

    def __repr__(self):
        return f"{self.valore} di {self.segno}"

class Giocatore:
    def __init__(self, id: int):
        self.id = id
        self.carta = None

    def ricevi_carta(self, carta: Carta):
        self.carta = carta

    def indovina(self, carte_avversari: list[Carta]):
        max_valore_avversari = max(c.valore for c in carte_avversari)
        scelta = random.choice([True, False]) # TODO
        assert self.carta is not None
        return scelta, ((self.carta.valore > max_valore_avversari) == scelta)

def distribuisci_carte(num_giocatori: int):
    mazzo = [Carta(valore) for valore in range(40)]
    random.shuffle(mazzo)
    return mazzo[:num_giocatori]

def gioca_partita(num_giocatori: int):
    giocatori = [Giocatore(id) for id in range(num_giocatori)]
    carte = distribuisci_carte(num_giocatori)

    for giocatore, carta in zip(giocatori, carte):
        giocatore.ricevi_carta(carta)

    risultati: list[dict[str, Any]] = []
    k = []
    for giocatore in giocatori:
        carte_avversari = [g.carta for g in giocatori if g != giocatore]
        vinco, indovinato = giocatore.indovina(carte_avversari)
        assert giocatore.carta is not None
        risultati.append({
            'giocatore': giocatore.id,
            'carta': giocatore.carta.id,
            'indovinato': indovinato,
            'vinco': vinco,
            'k': k.copy()
        })
        k.append(vinco)

    return risultati

def crea_database_partite(num_partite: int, num_giocatori: int):
    partite: list[list[dict[str, Any]]] = [[] for _ in range(num_giocatori)]
    for _ in range(num_partite):
        risultati = gioca_partita(num_giocatori)
        # [ { 'z': [12, 30, 2], 'k': [1, 1, -1], 'giocatore': 3, 'carta': 13, 'vinco': False, 'indovinato': True }, ... ]
        # 'z' indica le carte degli avversari
        # 'k' indica cosa han detto i giocatori prima di me
        # 'giocatore' indica il numero del turno
        # [ 'k', 'g', 'c', 'v', 'i' ]
        for ris in risultati:
            partite[ris['giocatore']].append({
                'z': [r['carta'] for r in risultati if r != ris],
                'k': ris['k'],
                'v': ris['vinco'],
                'i': ris['indovinato']
            })
    return [pd.DataFrame(esiti) for esiti in partite]

if __name__ == "__main__":
    num_partite = 10
    num_giocatori = 4
    list_of_df_partite = crea_database_partite(num_partite, num_giocatori)
    for i in range(len(list_of_df_partite)):
        df = list_of_df_partite[i]
        name = f"db_{num_giocatori}_{i}.csv"
        df.to_csv(f"dbs/{name}", index=False)
        print(f"Database salvato come {name}")

