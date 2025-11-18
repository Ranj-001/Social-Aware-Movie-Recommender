import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from bs4 import BeautifulSoup
import pickle
import requests
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import os
from collections import defaultdict

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

def create_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key' # Replace with a strong secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
# External API keys / config
# Prefer environment variable, but fall back to the same demo key used on the JS side
# so that poster images work out-of-the-box in local/dev setups.
app.config['TMDB_API_KEY'] = os.getenv('TMDB_API_KEY', '3611b01fcb811755c258eeb751e25b7b')
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    friends = db.relationship('Friendship', foreign_keys='Friendship.user_id', backref='user', lazy=True)
    friend_of = db.relationship('Friendship', foreign_keys='Friendship.friend_id', backref='friend', lazy=True)
    ratings = db.relationship('Rating', backref='rater', lazy=True)
    watch_history = db.relationship('WatchHistory', backref='user', lazy=True)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

class Friendship(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    friend_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date_established = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"Friendship('{self.user_id}' to '{self.friend_id}')"

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_title = db.Column(db.String(100), nullable=False)
    rating = db.Column(db.Integer, nullable=False) # e.g., 1-5 stars
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"Rating(User:{self.user_id}, Movie:'{self.movie_title}', Rating:{self.rating})"

class WatchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_title = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"WatchHistory(User:{self.user_id}, Movie:'{self.movie_title}')"

# Create database tables
with app.app_context():
    db.create_all()

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user, remember=True) # Remember me functionality
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html')

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/profile/<username>")
@login_required
def profile(username):
    user = User.query.filter_by(username=username).first_or_404()
    # For now, just display the profile. Later, add friends, ratings, etc.
    return render_template('profile.html', user=user)

@app.route("/add_friend", methods=['POST'])
@login_required
def add_friend():
    friend_username = request.form.get('friend_username')
    friend = User.query.filter_by(username=friend_username).first()

    if not friend:
        flash(f'User "{friend_username}" not found.', 'danger')
        return redirect(url_for('profile', username=current_user.username))

    if current_user.id == friend.id:
        flash('You cannot add yourself as a friend!', 'danger')
    elif Friendship.query.filter_by(user_id=current_user.id, friend_id=friend.id).first():
        flash(f'You are already friends with {friend.username}!', 'info')
    else:
        friendship = Friendship(user_id=current_user.id, friend_id=friend.id)
        db.session.add(friendship)
        # For a bidirectional friendship, add the inverse as well
        inverse_friendship = Friendship(user_id=friend.id, friend_id=current_user.id)
        db.session.add(inverse_friendship)
        db.session.commit()
        flash(f'You are now friends with {friend.username}!', 'success')
    return redirect(url_for('profile', username=current_user.username))

@app.route("/remove_friend/<int:friend_id>", methods=['POST'])
@login_required
def remove_friend(friend_id):
    friend = User.query.get_or_404(friend_id)
    friendship1 = Friendship.query.filter_by(user_id=current_user.id, friend_id=friend_id).first()
    friendship2 = Friendship.query.filter_by(user_id=friend_id, friend_id=current_user.id).first()

    if friendship1:
        db.session.delete(friendship1)
    if friendship2:
        db.session.delete(friendship2)
    
    if friendship1 or friendship2:
        db.session.commit()
        flash(f'You are no longer friends with {friend.username}.', 'success')
    else:
        flash(f'You were not friends with {friend.username}.', 'info')
    return redirect(url_for('profile', username=friend.username))


@app.route("/rate_movie/<movie_title>", methods=['POST'])
@login_required
def rate_movie(movie_title):
    rating_value = request.form.get('rating_value')
    if rating_value:
        rating = Rating(user_id=current_user.id, movie_title=movie_title, rating=int(rating_value))
        db.session.add(rating)
        db.session.commit()
        flash(f'You rated "{movie_title}" {rating_value} stars!', 'success')
    else:
        flash('Please provide a rating.', 'danger')
    # Redirect back to the movie details page or a confirmation page
    return redirect(url_for('home')) # Placeholder, ideally redirect to the movie's page

@app.route("/log_watch_history/<movie_title>", methods=['POST'])
@login_required
def log_watch_history(movie_title):
    watch_entry = WatchHistory(user_id=current_user.id, movie_title=movie_title)
    db.session.add(watch_entry)
    db.session.commit()
    flash(f'"{movie_title}" added to your watch history!', 'success')
    return redirect(url_for('home')) # Placeholder, ideally redirect to the movie's page


@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    
    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}
    print(f"calling imdb api: {'https://www.imdb.com/title/{}/reviews/?ref_=tt_ov_rt'.format(imdb_id)}")
    # web scraping to get user reviews from IMDB site
    url = f'https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'}

    response = requests.get(url, headers=headers, timeout=10)
    print(response.status_code)
    reviews_list = []
    reviews_status = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'lxml')
        # Try multiple selectors as IMDB changes often
        candidates = []
        candidates.extend(soup.select('div.ipc-html-content-inner-div'))
        candidates.extend(soup.select('div.review-container .text, div.content .text'))
        for node in candidates:
            text = (node.get_text(strip=True) or '').strip()
            if text:
                reviews_list.append(text)
                movie_vector = vectorizer.transform([text])
                pred = clf.predict(movie_vector)
                reviews_status.append('Good' if int(pred[0]) == 1 else 'Bad')
    else:
        print("Failed to retrieve reviews")

    # Build sentiment summary
    pos = sum(1 for s in reviews_status if s == 'Good')
    total = len(reviews_status)
    sentiment_summary = {
        'total': total,
        'positive': pos,
        'positive_rate': round((pos/total)*100, 1) if total else 0.0
    }

    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
            vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
            movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details,
            sentiment_summary=sentiment_summary)

# Heuristic social-aware personalized recommendations as a fallback
# If a GNN model is available later, we will switch to it inside this endpoint transparently.
@app.route('/api/home_recs')
@login_required
def api_home_recs():
    # Gather user signals
    user_ratings = Rating.query.filter_by(user_id=current_user.id).all()
    rated_titles = {r.movie_title: r.rating for r in user_ratings}

    # Candidate pool from: content-based similar to user's top-rated + friends' top-rated unseen
    # Load content data and similarity lazily
    global data, similarity
    try:
        data.head()
        _ = similarity.shape
    except Exception:
        data, similarity = create_similarity()

    def safe_rcmd(title):
        try:
            return rcmd(title) if title else []
        except Exception:
            return []

    # We treat this endpoint as the social/GNN-based recommendation endpoint.
    # Do NOT include cosine-similarity (content) based candidates here — those belong to the similarity endpoint.
    candidates = defaultdict(lambda: {'score': 0.0, 'reasons': set()})

    # Friends' top rated unseen
    friend_edges = Friendship.query.filter_by(user_id=current_user.id).all()
    friend_ids = [f.friend_id for f in friend_edges]
    if friend_ids:
        friends_ratings = Rating.query.filter(Rating.user_id.in_(friend_ids)).all()
        # Group by movie
        movie_to_friend_ratings = defaultdict(list)
        for r in friends_ratings:
            movie_to_friend_ratings[r.movie_title].append((r.user_id, r.rating))
        for mv, frs in movie_to_friend_ratings.items():
            if mv in rated_titles:
                continue
            # weight by rating and number of friends
            avg = sum(rt for _, rt in frs) / len(frs)
            candidates[mv]['score'] += 0.8 * (avg / 5.0) * min(len(frs), 3) / 3.0
            topf = sorted(frs, key=lambda x: -x[1])[0]
            friend_name = User.query.get(topf[0]).username if topf else 'friend'
            candidates[mv]['reasons'].add(f"Your friend {friend_name} rated it {topf[1]}")

    # If there are no friend-based candidates, return an empty recommendation list
    # (do not inject cosine-similarity content into the social/GNN feed).

    # Sentiment prior using previously computed reviews cache would be best; for now, boost by dataset 'comb' contains text, no sentiment.
    # Optional: small boost based on global vote_average from TMDB could be added client-side.

    # Rank and take top-K
    ranked = sorted(candidates.items(), key=lambda x: -x[1]['score'])[:12]

    # Normalize into response
    recs = []
    for title, meta in ranked:
        # Try to enrich with poster image from TMDB when available. Use a local static
        # placeholder by default so that we always show an image even if TMDB is unreachable
        # or no API key is configured.
        poster_url = url_for('static', filename='image.jpg')
        tmdb_key = app.config.get('TMDB_API_KEY', '')
        if tmdb_key:
            try:
                resp = requests.get(
                    'https://api.themoviedb.org/3/search/movie',
                    params={'api_key': tmdb_key, 'query': title},
                    timeout=5,
                )
                if resp.status_code == 200:
                    results = resp.json().get('results', [])
                    if results:
                        p = results[0].get('poster_path')
                        if p:
                            poster_url = f'https://image.tmdb.org/t/p/w500{p}'
            except Exception:
                # network errors or timeouts -> fall back to the local placeholder
                pass

        recs.append({
            'title': title.title(),
            'score': round(meta['score'], 3),
            'reasons': list(meta['reasons'])[:2],
            'poster': poster_url
        })

    return jsonify({'recommendations': recs})

if __name__ == '__main__':
    app.run(debug=True)
