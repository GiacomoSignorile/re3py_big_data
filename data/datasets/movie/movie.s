[Relations]
movies_category(MovieId, nominalTarget)
movies_title(MovieId, nominal0)
movies_release_date(MovieId, numeric2)
movies_url(MovieId, nominal4)
nation_n_name(NationId, nominal5)
nation_n_regionkey(NationId, RegionId)
nation_n_comment(NationId, nominal6)
ratings_id_movie(UserId, MovieId)
ratings_rating(UserId, numeric7)
ratings_date(UserId, numeric8)
region_r_name(RegionId, nominal9)
region_r_comment(RegionId, nominal102)
users_age(UserId, numeric11)
users_gender(UserId, nominal12)
users_nation(UserId, NationId)
users_occupation(UserId, nominal13)
[Aggregates]
sum
count
max
min
mean
[AtomTests]
movies_title(old, new)
movies_release_date(old, new)
//movies_url(MovieId, nominal4)
nation_n_name(old, new)
nation_n_regionkey(old, new)
//nation_n_comment(old, new)
ratings_id_movie(new, old)
ratings_id_movie(old, new)
ratings_rating(old, new)
//ratings_date(old, new)
region_r_name(old, new)
region_r_comment(old, new)
users_age(old, new)
users_gender(old, new)
users_nation(old, new)
users_occupation(old, new)
