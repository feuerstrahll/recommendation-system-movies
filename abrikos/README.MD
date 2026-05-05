### План действий
- Сначала:
> python clean_movies_data.py --input ./data_Movies --output ./cleaned_data
- Потом:
> python load_to_postgres.py `
>         --host localhost --port 5432 `
>         --db movies_db --user postgres --password secret `   
>         --data ./cleaned_data
*Можно и в одну строчку как я сделал дальше*
- И в конце:
> python check_db.py --password secret --db movies_db
