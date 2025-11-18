document.addEventListener('DOMContentLoaded', function () {
  try {
    // Expect a global `films` array injected from Flask: var films = {{ suggestions|tojson|safe }};
    var titles = Array.isArray(window.films) ? window.films.slice() : [];
    if (!Array.isArray(titles)) {
      console.warn('autocomplete: window.films is not an array; value =', window.films);
      titles = [];
    }

    // Normalize to plain strings and trim
    titles = titles.map(function (t) {
      if (typeof t !== 'string') return String(t == null ? '' : t);
      return t.trim();
    }).filter(function (t) { return !!t; });

    var input = document.getElementById('autoComplete');
    if (!input) {
      console.warn('autocomplete: #autoComplete input not found');
      return;
    }

    // Create / reuse the dropdown container right after the input
    var existing = document.getElementById('autoComplete_list');
    var list = existing || document.createElement('ul');
    list.id = 'autoComplete_list';
    list.className = 'autoComplete_result';
    list.style.display = 'none';
    if (!existing) {
      if (input.parentNode) {
        input.parentNode.insertBefore(list, input.nextSibling);
      } else {
        document.body.appendChild(list);
      }
    }

    var highlightedIndex = -1;

    function clearList() {
      list.innerHTML = '';
      list.style.display = 'none';
      highlightedIndex = -1;
    }

    function renderList(matches) {
      clearList();
      if (!matches.length) {
        var li = document.createElement('li');
        li.className = 'no_result';
        li.textContent = 'No Results';
        list.appendChild(li);
        list.style.display = 'block';
        return;
      }

      matches.forEach(function (title, idx) {
        var li = document.createElement('li');
        li.textContent = title;
        li.setAttribute('data-index', String(idx));
        li.addEventListener('mousedown', function (e) {
          // mousedown fires before blur
          e.preventDefault();
          selectIndex(idx);
        });
        list.appendChild(li);
      });
      list.style.display = 'block';
    }

    function selectIndex(idx) {
      var items = list.querySelectorAll('li');
      if (!items.length || idx < 0 || idx >= items.length) return;
      var value = items[idx].textContent || '';
      input.value = value;
      clearList();
      var btn = document.getElementById('enterBtn');
      if (btn) btn.disabled = !input.value.trim();
    }

    function updateHighlight(newIndex) {
      var items = list.querySelectorAll('li');
      for (var i = 0; i < items.length; i++) {
        if (i === newIndex) {
          items[i].classList.add('active');
        } else {
          items[i].classList.remove('active');
        }
      }
      highlightedIndex = newIndex;
    }

    function handleInput() {
      var query = (input.value || '').toLowerCase().trim();
      var btn = document.getElementById('enterBtn');
      if (btn) btn.disabled = !query;

      if (!query) {
        clearList();
        return;
      }

      // Simple substring match, case-insensitive
      var matches = [];
      for (var i = 0; i < titles.length && matches.length < 8; i++) {
        var t = titles[i];
        if (t.toLowerCase().indexOf(query) !== -1) {
          matches.push(t);
        }
      }
      renderList(matches);
    }

    input.addEventListener('input', function () {
      // debounce lightly using requestAnimationFrame
      if (window.requestAnimationFrame) {
        window.requestAnimationFrame(handleInput);
      } else {
        handleInput();
      }
    });

    input.addEventListener('keydown', function (e) {
      var items = list.querySelectorAll('li');
      if (!items.length || list.style.display === 'none') return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        var next = highlightedIndex + 1;
        if (next >= items.length) next = 0;
        updateHighlight(next);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        var prev = highlightedIndex - 1;
        if (prev < 0) prev = items.length - 1;
        updateHighlight(prev);
      } else if (e.key === 'Enter') {
        if (highlightedIndex >= 0) {
          e.preventDefault();
          selectIndex(highlightedIndex);
        }
      } else if (e.key === 'Escape') {
        clearList();
      }
    });

    // Hide list when clicking outside
    document.addEventListener('click', function (e) {
      if (e.target !== input && !list.contains(e.target)) {
        clearList();
      }
    });

  } catch (err) {
    console.error('autocomplete: initialization error', err);
  }
});
