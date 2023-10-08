using Dates

hfun_todo(args) = """<text style="color: red; font-style: italic;">$(join(args, " "))</text>"""

function hfun_include(params)
    rpath = params[1]
    fullpath = joinpath(Franklin.path(:folder), rpath)
    read(fullpath, String)
end

format_date(d) = Dates.format(d, Franklin.globvar("date_format"))
short_format_date(d) = Dates.format(d, Franklin.globvar("short_date_format"))

function hfun_format_date(params)
    datevar, = params
    format_date(Franklin.locvar(datevar))
end

function hfun_short_format_date(params)
    datevar, = params
    short_format_date(Franklin.locvar(datevar))
end

function filter_files(flt, path)
    dirs = walkdir(path)
    files = map(((root, a, files), ) -> map(f -> joinpath(root, f), files), dirs)
    files = Iterators.flatten(files)
    Iterators.filter(flt, files)
end

# trim the (/index).md off
path2url(f) = replace(f, r"(/index\.md)|(\.md)" => "")

struct PageListItem
    title::String
    url::String
    date::Date
end

Base.isless(p1::PageListItem, p2::PageListItem) = p1.date < p2.date

function Base.show(io::IO, p::PageListItem)
    write(io, """<div class="post-title">""")
    write(io, """<span class="post-date">$(short_format_date(p.date))</span>""")
    write(io, """<div class="flex-break"></div>""")
    write(io, """<a class="post-link" href="$(p.url)">$(p.title)</a>""")
    write(io, """</div>""")
end

function style_page_list(pages)
    io = IOBuffer()
    for page in pages
        show(io, page)
    end
    String(take!(io))
end

function hfun_list_pages(params)
    rpath = params[1]
    fullpath = joinpath(Franklin.path(:folder), rpath)
    pages = PageListItem[]
    for f in filter_files(endswith(".md"), fullpath)
        title = Franklin.pagevar(f, :title)
        date = Franklin.pagevar(f, :date)
        url = relpath(f, fullpath) |> path2url
        p = PageListItem(title, url, date)
        push!(pages, p)
    end
    pages = sort(pages, rev = true)
    if length(params) > 1 # only take n most recent pages
        pages = Iterators.take(pages, parse(Int, params[2]))
    end
    style_page_list(pages)
end

include("bib.jl")
