# Taxonomie politique

## Rôle

La taxonomie fournit un vocabulaire commun pour organiser les questions, les passages et les sorties. Elle ne remplace pas la preuve documentaire et ne doit pas forcer un texte dans une catégorie absente du document.

## Taxonomie de démonstration

| Thème | Contenu indicatif |
| --- | --- |
| économie | emploi, salaires, investissement, industrie, fiscalité, inflation |
| souveraineté | indépendance, frontières, autonomie, autodétermination |
| démocratie | élections, transparence, institutions, droits, participation |
| politique sociale | école, santé, logement, jeunesse, protection sociale |
| environnement | climat, énergie, eau, terres, transition écologique |

Cette couche compacte est utilisée uniquement par la démo publique.

## Registre de recherche

L'architecture réelle s'appuie sur les fiches YAML de `registry/`. Chaque fiche précise notamment :

- la question et la définition de la variable ;
- le type de sortie ;
- l'échelle ou les labels autorisés ;
- les régimes de preuve acceptés ;
- les critères d'inclusion et d'exclusion ;
- les règles de décision et les cas ambigus.

## Extension

Une extension de la taxonomie doit être versionnée, testée sur des exemples positifs et négatifs et validée avant son utilisation dans le raisonnement final.
