-- The website's Supabase project: run once in the project's SQL Editor.
-- Accounts themselves are Supabase Auth users (email + password). Each account's dashboard settings
-- and agreement record are one row here, which only that account can read or change.

-- Each user's saved settings and agreement record
create table public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  prefs jsonb not null default '{}'::jsonb,
  agreed jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint prefs_small check (pg_column_size(prefs) < 16384),
  constraint agreed_small check (agreed is null or pg_column_size(agreed) < 1024)
);

-- Row Level Security: each person can only see and change their own row
alter table public.profiles enable row level security;
create policy "own profile: read" on public.profiles
  for select to authenticated using ((select auth.uid()) = id);
create policy "own profile: create" on public.profiles
  for insert to authenticated with check ((select auth.uid()) = id);
create policy "own profile: change" on public.profiles
  for update to authenticated using ((select auth.uid()) = id) with check ((select auth.uid()) = id);

-- "Delete account": lets a signed-in user delete only themselves (their row goes with them)
create function public.delete_my_account() returns void
language sql security definer set search_path = ''
as $$ delete from auth.users where id = auth.uid(); $$;
revoke execute on function public.delete_my_account() from public, anon;
grant execute on function public.delete_my_account() to authenticated;
